import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_categorical = tf.random.categorical


def random_categorical(*args, **kwargs):
    kwargs["seed"] = None
    return _orig_random_categorical(*args, **kwargs)


tf.random.categorical = random_categorical

# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_normal = tf.random.normal


def random_normal(*args, **kwargs):
    kwargs["seed"] = None
    return _orig_random_normal(*args, **kwargs)


tf.random.normal = random_normal


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return samples.mean(0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -logprob.mean(0)


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=None):
        self._sample_dtype = dtype or tf.float32
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        return tf.cast(super().mode(), self._sample_dtype)

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
        return sample

    def _pad(self, tensor, shape):
        tensor = super().probs_parameter()
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor


class TruncNormalDist(tfd.TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

    def sample(self, *args, **kwargs):
        event = super().sample(*args, **kwargs)
        if self._clip:
            clipped = tf.clip_by_value(
                event, self.low + self._clip, self.high - self._clip
            )
            event = event - tf.stop_gradient(event) + tf.stop_gradient(clipped)
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.0),
            tf.clip_by_value(y, -0.99999997, 0.99999997),
            y,
        )
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


class MSEDist:
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - symlog(value)) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


def symlog(x):
    return tf.sign(x) * tf.math.log(1 + tf.abs(x))


def symexp(x):
    return tf.sign(x) * (tf.math.exp(tf.abs(x)) - 1)
