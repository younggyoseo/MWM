import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common


class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.act_space, self.tfstep)
        self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)

    @tf.function
    def policy(self, obs, state=None, mode="train"):
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(
            lambda: self.tfstep.assign(int(self.step), read_value=False), [], []
        )
        if state is None:
            latent = self.wm.rssm.initial(len(obs["reward"]))
            action = tf.zeros((len(obs["reward"]),) + self.act_space.shape)
            state = latent, action
        latent, action = state

        # Extract autoencoder representation
        data = self.wm.preprocess(obs)
        mae_latent, mask, ids_restore = self.wm.mae_encoder.forward_encoder(
            data["image"], 0.0, 1
        )
        ## Move [CLS] to last position for positional embedding
        mae_latent = tf.concat([mae_latent[:, 1:], mae_latent[:, :1]], axis=1)
        wm_latent = self.wm.wm_vit_encoder.forward_encoder(mae_latent)
        embed = wm_latent.mean(1)

        # RSSM one step forward
        sample = (mode == "train") or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], sample
        )
        feat = self.wm.rssm.get_feat(latent)

        # Policy
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = common.action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = (latent, action)
        return outputs, state

    @tf.function
    def train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.train(data, self._task_behavior.actor, state)
        metrics.update(mets)
        start = outputs["post"]
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode()
        metrics.update(
            self._task_behavior.train(self.wm, start, data["is_terminal"], reward)
        )
        return state, metrics

    @tf.function
    def train_mae(self, data):
        metrics = {}
        mets = self.wm.train_mae(data)
        metrics.update(mets)
        return metrics

    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        report["openl_image"] = self.wm.video_pred(data)
        return report


class WorldModel(common.Module):
    def __init__(self, config, obs_space, act_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.act_space = act_space

        # RSSM
        self.rssm = common.RSSM(**config.rssm)
        self.feat_dim = config.rssm.deter + config.rssm.stoch * config.rssm.discrete

        # MAE
        self.mae_encoder, self.mae_decoder = common.mae_factory(**config.mae)

        # ViT for latent dynamics model
        self.wm_vit_encoder, self.wm_vit_decoder = common.flat_vit_factory(
            **config.wm_flat_vit
        )

        # Heads in latent dynamics model
        self.heads = {}
        self.heads["reward"] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads["discount"] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name

        # Optimizers
        self.model_opt = common.Optimizer("model", **config.model_opt)
        self.mae_opt = common.Optimizer("mae", **config.mae_opt)

        # ImageNet stats
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406])
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225])

        self.step = 0

    def train(self, data, actor, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(
                data, actor, state, training=True
            )
        modules = [
            self.rssm,
            self.wm_vit_encoder,
            self.wm_vit_decoder,
            *self.heads.values(),
        ]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def train_mae(self, data):
        with tf.GradientTape() as mae_tape:
            mae_loss, metrics = self.loss_mae(data, training=True)
        modules = [
            self.mae_encoder,
            self.mae_decoder,
        ]
        metrics.update(self.mae_opt(mae_tape, mae_loss, modules))
        return metrics

    def loss(self, data, actor, state=None, training=False):
        data = self.preprocess(data)
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        likes, losses, metrics = {}, {}, {}

        # Forward without masking
        m = 0.0
        latent, mask, _ = self.mae_encoder.forward_encoder(videos, m, T)
        feature = latent
        data["feature"] = tf.stop_gradient(feature.astype(tf.float32))

        # Detach features
        feature = tf.stop_gradient(feature)

        # ViT encoder with average pooling
        ## Move [CLS] to last position
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature)
        embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # RSSM forward
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        feat = self.rssm.get_feat(post)
        kl_loss = kl_value = self.rssm.kl_loss(post, prior, self.config.wmkl_balance)
        losses["kl"] = tf.clip_by_value(
            kl_loss * self.config.wmkl.scale, self.config.wmkl_minloss, 100.0
        ).mean()

        # Non-image losses
        dists = {}
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        for key, dist in dists.items():
            like = tf.cast(dist.log_prob(data[key]), tf.float32)
            likes[key] = like
            losses[key] = -like.mean()

        # Feature reconstruction loss
        feat = tf.reshape(feat, [B * T, 1, self.feat_dim])
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)
        dist = common.MSEDist(tf.cast(feature_pred, tf.float32), 1, "sum")
        like = tf.cast(dist.log_prob(data["feature"]), tf.float32)
        likes["feature"] = like
        losses["feature"] = -like.mean()

        # Summation and log metrics
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def loss_mae(self, data, training=False):
        data = self.preprocess(data)
        key = "image"
        videos = data[key]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        losses, metrics = {}, {}

        # MAE forward
        m = self.config.mask_ratio
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, 1)

        if self.config.mae.reward_pred:
            decoder_pred, reward_pred = self.mae_decoder.forward_decoder(
                latent, ids_restore
            )
            # Reward prediction loss
            reward_pred = tf.reshape(reward_pred, [B, T, 1])
            reward = tf.reshape(data["reward"], [B, T, 1])
            reward_loss = self.mae_decoder.forward_reward_loss(reward, reward_pred)
            losses["mae_reward"] = reward_loss
        else:
            decoder_pred = self.mae_decoder.forward_decoder(latent, ids_restore)

        # Image reconstruction loss
        decoder_loss = self.mae_decoder.forward_loss(videos, decoder_pred, mask)
        losses[key] = decoder_loss

        # Summation and log metrics
        mae_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        return mae_loss, metrics

    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start["feat"] = self.rssm.get_feat(start)
        start["action"] = tf.zeros_like(policy(start["feat"]).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(tf.stop_gradient(seq["feat"][-1])).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = tf.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * tf.ones(seq["feat"].shape[:-1])
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = tf.math.cumprod(
            tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:  # image
                value = self.standardize(value.astype(dtype) / 255.0)
            obs[key] = value
        if self.config.clip_rewards in ["identity", "sign", "tanh"]:
            obs["reward"] = {
                "identity": tf.identity,
                "sign": tf.sign,
                "tanh": tf.tanh,
            }[self.config.clip_rewards](obs["reward"])
        else:
            obs["reward"] /= float(self.config.clip_rewards)
        obs["discount"] = 1.0 - obs["is_terminal"].astype(dtype)
        obs["discount"] *= self.config.discount
        return obs

    @tf.function
    def standardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = (x - mean) / std
        return x

    @tf.function
    def destandardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = x * std + mean
        return x

    @tf.function
    def video_pred(self, data):
        data = {k: v[:6] for k, v in data.items()}
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])

        # Autoencoder reconstruction
        m = 0.0 if self.config.mae.early_conv else self.config.mask_ratio
        recon_latent, recon_mask, recon_ids_restore = self.mae_encoder.forward_encoder(
            videos, m, T
        )
        recon_model = self.mae_decoder.forward_decoder(recon_latent, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon_model = recon_model[0]  # first element is decoded one
        recon_model = tf.cast(recon_model, tf.float32)
        recon_model = self.mae_decoder.unpatchify(recon_model[: B * T])
        recon_model = tf.cast(
            self.destandardize(recon_model.reshape([B, T, H, W, C])), tf.float32
        )

        # Latent dynamics model prediction
        # 1: Extract MAE representations
        m = 0.0
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, T)
        feature = tf.stop_gradient(latent)

        # 2: Reconstructions from conditioning frames
        # 2-1: Process through ViT encoder
        ## Move [CLS] to last position for positional embedding
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature)
        embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # 2-2: Process these through RSSM
        states, _ = self.rssm.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        feat = self.rssm.get_feat(states)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, self.feat_dim])

        # 2-3: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 2-4 Process these through MAE decoder
        recon_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, :5].reshape(
            [b * t, -1]
        )
        recon = self.mae_decoder.forward_decoder(feature_pred, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon = recon[0]
        recon = tf.cast(recon, tf.float32)
        recon = self.mae_decoder.unpatchify(recon[: b * t])
        recon = tf.reshape(
            recon, [b, t, recon.shape[1], recon.shape[2], recon.shape[3]]
        )
        recon = self.destandardize(recon)

        # 3: Open-loop prediction
        # 3-1: Process through RSSM to obtain prior
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        feat = self.rssm.get_feat(prior)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, self.feat_dim])

        # 3-2: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 3-3: Process these through MAE decoder
        openl_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, 5:].reshape(
            [b * t, -1]
        )
        openl = self.mae_decoder.forward_decoder(feature_pred, openl_ids_restore)
        if self.config.mae.reward_pred:
            openl = openl[0]
        openl = tf.cast(openl, tf.float32)
        openl = self.mae_decoder.unpatchify(openl[: b * t])
        openl = tf.reshape(
            openl, [b, t, openl.shape[1], openl.shape[2], openl.shape[3]]
        )
        openl = self.destandardize(openl)

        # Concatenate across timesteps
        model = tf.concat([recon, openl], 1)
        truth = tf.cast(
            self.destandardize(videos.reshape([B, T, H, W, C])[:6]), tf.float32
        )
        video = tf.concat([truth, recon_model, model], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):
    def __init__(self, config, act_space, tfstep):
        self.config = config
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update(
                {"actor.dist": "onehot" if discrete else "trunc_normal"}
            )
        if self.config.actor_grad == "auto":
            self.config = self.config.update(
                {"actor_grad": "reinforce" if discrete else "dynamics"}
            )
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.critic = common.MLP([], **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer("actor", **self.config.actor_opt)
        self.critic_opt = common.Optimizer("critic", **self.config.critic_opt)
        self.ext_rewnorm = common.StreamNorm(**self.config.reward_norm)

    def train(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape() as actor_tape:
            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            reward = reward_fn(seq)
            reward, mets1 = self.ext_rewnorm(reward)
            mets1 = {f"reward_{k}": v for k, v in mets1.items()}
            seq["reward"] = reward
            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)
        with tf.GradientTape() as critic_tape:
            critic_loss, mets4 = self.critic_loss(seq, target)
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq["feat"][:-2]))
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        objective += self.config.aent.scale * ent
        weight = tf.stop_gradient(seq["weight"])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_ent"] = ent.mean()
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq["weight"])
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {"critic": dist.mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = tf.cast(seq["reward"], tf.float32)
        disc = tf.cast(seq["discount"], tf.float32)
        value = self._target_critic(seq["feat"]).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0,
        )
        metrics = {}
        metrics["critic_slow"] = value.mean()
        metrics["critic_target"] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = (
                    1.0
                    if self._updates == 0
                    else float(self.config.slow_target_fraction)
                )
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
