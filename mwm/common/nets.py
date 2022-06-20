import re
import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras import initializers as tfki
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):
    def __init__(
        self,
        action_free=False,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()
        self._action_free = action_free
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        return state

    def fill_action_with_zero(self, action):
        # action: [B, action]
        B, D = action.shape[0], action.shape[1]
        if self._action_free:
            return self._cast(tf.zeros([B, 50]))
        else:
            zeros = self._cast(tf.zeros([B, 50 - D]))
            return tf.concat([action, zeros], axis=1)

    @tf.function
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state[f"deter"]], -1)

    def get_dist(self, state):
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if is_first.any():
        prev_state, prev_action = tf.nest.map_structure(
            lambda x: tf.einsum("b,b...->b...", 1.0 - is_first.astype(x.dtype), x),
            (prev_state, prev_action),
        )
        prior = self.img_step(prev_state, prev_action, sample)
        x = tf.concat([prior[f"deter"], embed], -1)
        x = self.get("obs_out", tfkl.Dense, self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior[f"deter"], **stats}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, self.fill_action_with_zero(prev_action)], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x, deter = self._cell(x, [prev_state[f"deter"]])
        deter = deter[0]
        x = self.get("img_out", tfkl.Dense, self._hidden)(x)
        x = self.get("img_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer(f"img_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, balance=0.8):
        post_const = tf.nest.map_structure(tf.stop_gradient, post)
        prior_const = tf.nest.map_structure(tf.stop_gradient, prior)
        lhs = tfd.kl_divergence(self.get_dist(post_const), self.get_dist(prior))
        rhs = tfd.kl_divergence(self.get_dist(post), self.get_dist(prior_const))
        return balance * lhs + (1 - balance) * rhs


class MLP(common.Module):
    def __init__(
        self, shape, layers=[512, 512, 512, 512], act="elu", norm="none", **out
    ):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index, unit in enumerate(self._layers):
            x = self.get(f"dense{index}", tfkl.Dense, unit)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=True, act="tanh", update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, **kwargs)
        if norm:
            self._norm = NormLayer("layer")
        else:
            self._norm = NormLayer("none")

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        parts = self._norm(parts)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class DistLayer(common.Module):
    def __init__(self, shape, dist="mse", outscale=0.1, min_std=0.1, max_std=1.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._max_std = max_std
        self._outscale = outscale

    def __call__(self, inputs):
        kw = {}
        if self._outscale == 0.0:
            kw["kernel_initializer"] = tfki.Zeros()
        else:
            kw["kernel_initializer"] = tfki.VarianceScaling(
                self._outscale, "fan_avg", "uniform"
            )
        out = self.get("out", tfkl.Dense, np.prod(self._shape), **kw)(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == "mse":
            return common.MSEDist(out, len(self._shape), "sum")
        if self._dist == "symlog":
            return common.SymlogDist(out, len(self._shape), "sum")
        if self._dist == "nmse":
            return common.NormalizedMSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._min_std, self._max_std
            std = (hi - lo) * tf.nn.sigmoid(std) + lo
            dist = tfd.Normal(tf.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "trunc_normal":
            lo, hi = self._min_std, self._max_std
            std = (hi - lo) * tf.nn.sigmoid(std) + lo
            dist = tfd.TruncatedNormal(tf.tanh(out), std, -1, 1)
            dist = tfd.Independent(dist, 1)
            dist.minent = np.prod(self._shape) * tfd.Normal(0.99, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "onehot":
            dist = common.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * np.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)

class NormLayer(common.Module, tf.keras.layers.Layer):
    def __init__(self, impl):
        super().__init__()
        self._impl = impl

    def build(self, input_shape):
        if self._impl == "keras":
            self.layer = tfkl.LayerNormalization()
            self.layer.build(input_shape)
        elif self._impl == "layer":
            self.scale = self.add_weight("scale", input_shape[-1], tf.float32, "Ones")
            self.offset = self.add_weight(
                "offset", input_shape[-1], tf.float32, "Zeros"
            )

    def call(self, x):
        if self._impl == "none":
            return x
        elif self._impl == "keras":
            return self.layer(x)
        elif self._impl == "layer":
            mean, var = tf.nn.moments(x, -1, keepdims=True)
            return tf.nn.batch_normalization(
                x, mean, var, self.offset, self.scale, 1e-3
            )
        else:
            raise NotImplementedError(self._impl)


class MLPEncoder(common.Module):
    def __init__(
        self, act="elu", norm="none", layers=[512, 512, 512, 512], batchnorm=False
    ):
        self._act = get_act(act)
        self._layers = layers
        self._norm = norm
        self._batchnorm = batchnorm

    @tf.function
    def __call__(self, x, training=False):
        x = x.astype(prec.global_policy().compute_dtype)
        if self._batchnorm:
            x = self.get(f"batchnorm", tfkl.BatchNormalization)(x, training=training)
        for i, unit in enumerate(self._layers):
            x = self.get(f"dense{i}", tfkl.Dense, unit)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class CNNEncoder(common.Module):
    def __init__(
        self,
        cnn_depth=64,
        cnn_kernels=(4, 4),
        act="elu",
    ):
        self._act = get_act(act)
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

    @tf.function
    def __call__(self, x):
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2**i * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2D, depth, kernel, 1)(x)
            x = self._act(x)
        return x


class CNNDecoder(common.Module):
    def __init__(
        self,
        out_dim,
        cnn_depth=64,
        cnn_kernels=(4, 5),
        act="elu",
    ):
        self._out_dim = out_dim
        self._act = get_act(act)
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

    @tf.function
    def __call__(self, x):
        x = x.astype(prec.global_policy().compute_dtype)

        x = self.get("convin", tfkl.Dense, 2 * 2 * 2 * self._cnn_depth)(x)
        x = tf.reshape(x, [-1, 1, 1, 8 * self._cnn_depth])

        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 1) * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2DTranspose, depth, kernel, 1)(x)
            x = self._act(x)
        x = self.get("convout", tfkl.Dense, self._out_dim)(x)
        return x


def get_act(name):
    if name == "none":
        return tf.identity
    if name == "mish":
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)
