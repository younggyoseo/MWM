import pickle
import tfimm

from tfimm.architectures.vit import ViTBlock
from tfimm.layers import PatchEmbeddings
from tfimm.layers.factory import norm_layer_factory

import common

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers as tfkl
from tensorflow.keras import mixed_precision as prec


class MaskedViTEncoder(common.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = int((img_size // patch_size) ** 2)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.early_conv = early_conv
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                embed_dim,
                int(img_size // patch_size),
                cls_token=True,
                add_token=False,
            )[None],
            name="encoder_pos_embed",
            dtype=tf.float32,
        )

    def patchify(self, imgs):
        """
        imgs: [N, H, W, 3]
        x: [N, L, patch_size**2 * 3]
        """
        p = self.patch_size
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0

        x = tf.image.extract_patches(
            imgs, [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID"
        )
        x = tf.reshape(x, [imgs.shape[0], -1, p ** 2 * imgs.shape[-1]])

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, H, W, 3)
        """
        p = self.patch_size
        c = x.shape[-1] // (p ** 2)
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = tf.reshape(x, [x.shape[0], h, w, p, p, c])
        x = tf.einsum("nhwpqc->nhpwqc", x)
        imgs = tf.reshape(x, [x.shape[0], h * p, h * p, c])
        return imgs

    def random_tube_masking(self, x, mask_ratio, T):
        # If T=1, random masking

        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        B = N // T
        assert B * T == N

        # generate noise - consistent over timesteps
        noise = tf.random.uniform([B, L], 0.0, 1.0)
        if mask_ratio == 0.0:
            # important to avoid shuffling when m == 0
            noise = tf.sort(noise)
        noise = tf.repeat(tf.expand_dims(noise, 1), repeats=T, axis=1)
        noise = tf.reshape(noise, [N, L])

        # sort noise for each sample
        # keep small, remove large
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # trick for tensorflow-gather
        row_ids = tf.ones_like(ids_shuffle) * tf.expand_dims(tf.range(N), 1)
        _ids_shuffle = tf.stack([row_ids, ids_shuffle], -1)  # [N, L, 2]
        _ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        # keep the first subset
        ids_keep = _ids_shuffle[:, :len_keep]
        x_masked = tf.gather_nd(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = tf.concat([tf.zeros([N, len_keep]), tf.ones([N, L - len_keep])], axis=1)
        # unshuffle to get ther binary mask
        mask = tf.gather_nd(mask, _ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, T):
        # embed patches
        x = self._cast(x)
        batch_size = tf.shape(x)[0]

        if self.early_conv:
            for i in range(3):
                depth = self.embed_dim // (2 ** (3 - i))
                x = self.get(
                    f"early_conv_{i}",
                    tfkl.Conv2D,
                    depth,
                    4,
                    2,
                    padding="SAME",
                )(x)
                x = tf.nn.relu(x)
            x = self.get("early_conv_proj", tfkl.Conv2D, self.embed_dim, 1, 1)(x)
            x = tf.reshape(x, [x.shape[0], -1, self.embed_dim])
        else:
            x = self.get(
                "encoder_patch_embed",
                PatchEmbeddings,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                norm_layer="",
            )(x)

        # add pos embed w/o cls token
        x = x + self._cast(self.pos_embed[:, 1:, :])

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_tube_masking(x, mask_ratio, T)

        # append class token
        cls_token = self.get("cls_token", common.mae_utils.ClsToken, self.embed_dim)(x)
        cls_token = cls_token + self.pos_embed[:, :1, :]
        cls_tokens = tf.repeat(cls_token, repeats=batch_size, axis=0)
        x = tf.concat([self._cast(cls_tokens), x], axis=1)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x, mask, ids_restore


class MaskedViTDecoder(common.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        norm_pix_loss=False,
        masked_decoder_loss=False,
        reward_pred=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = int((img_size // patch_size) ** 2)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss
        self.masked_decoder_loss = masked_decoder_loss
        self.reward_pred = reward_pred
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.decoder_pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                decoder_embed_dim,
                int(img_size // patch_size),
                cls_token=True,
                add_token=reward_pred,
            )[None],
            name="decoder_pos_embed",
            dtype=tf.float32,
        )

    def patchify(self, imgs):
        """
        imgs: [N, H, W, 3]
        x: [N, L, patch_size**2 * 3]
        """
        p = self.patch_size
        c = imgs.shape[-1]
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0

        x = tf.image.extract_patches(
            imgs, [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID"
        )
        x = tf.reshape(x, [imgs.shape[0], -1, p ** 2 * c])

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, H, W, 3)
        """
        p = self.patch_size
        c = x.shape[-1] // (p ** 2)
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = tf.reshape(x, [x.shape[0], h, w, p, p, c])
        x = tf.einsum("nhwpqc->nhpwqc", x)
        imgs = tf.reshape(x, [x.shape[0], h * p, h * p, c])
        return imgs

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.decoder_embed_dim,
        )(x)

        # trick for tensorflow-gather
        N = ids_restore.shape[0]
        row_ids = tf.ones_like(ids_restore) * tf.expand_dims(tf.range(N), 1)
        ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        mask_token = self.get(
            "mask_token", common.mae_utils.MaskToken, self.decoder_embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1],
            )
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_ = tf.gather_nd(x_, ids_restore)  # unshuffle

        # append mask token for reward prediction
        # we use same mask token for rew prediction. Maybe try different token?
        if self.reward_pred:
            rew_mask_token = self._cast(
                tf.tile(
                    mask_token,
                    [x.shape[0], 1, 1],
                )
            )
            x_ = tf.concat([x_, rew_mask_token], axis=1)
        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x = x + tf.repeat(
            self._cast(self.decoder_pos_embed), repeats=x.shape[0], axis=0
        )

        # apply Transformer blocks
        for j in range(self.decoder_depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                ViTBlock,
                embed_dim=self.decoder_embed_dim,
                nb_heads=self.decoder_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        if self.reward_pred:
            # predictor projection
            x = self.get(
                "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
            )(x[:, 1:-1, :])
            # reward projection
            y = self.get("vit_reward_pred", tfkl.Dense, 1)(x[:, -1:, :])
            return x, y
        else:
            # predictor projection
            x = self.get(
                "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
            )(x[:, 1:, :])
            return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, H, W, 3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        imgs = tf.cast(imgs, tf.float32)
        pred = tf.cast(pred, tf.float32)
        mask = tf.cast(mask, tf.float32)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = tf.reduce_mean(target, axis=-1, keepdims=True)
            var = tf.reduce_var(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, -1)  # [N, L], mean loss per patch

        if self.masked_decoder_loss:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def forward_reward_loss(self, rews, preds):
        rews = tf.cast(rews, tf.float32)
        preds = tf.cast(preds, tf.float32)
        dist = common.SymlogDist(preds, 1, "mean")
        loss = -dist.log_prob(rews)
        return loss.mean()


class ViTEncoder(common.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = int((img_size // patch_size) ** 2)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                embed_dim, int(img_size // patch_size), cls_token=False, add_token=True
            )[None],
            name="encoder_pos_embed",
            dtype=tf.float32,
        )

    def forward_encoder(self, x):
        # embed patches
        x = self._cast(x)
        batch_size = tf.shape(x)[0]
        x = self.get("encoder_embed", tfkl.Dense, self.embed_dim)(x)
        x = x + self._cast(self.pos_embed)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x


class ViTDecoder(common.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = int((img_size // patch_size) ** 2) + 1
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.decoder_pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                decoder_embed_dim,
                int(img_size // patch_size),
                cls_token=True,
                add_token=True,  # one additional token to handle [CLS] token of MAE
            )[None],
            name="decoder_pos_embed",
            dtype=tf.float32,
        )

    def forward_decoder(self, x):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.decoder_embed_dim,
        )(x)

        mask_token = self.get(
            "mask_token", common.mae_utils.MaskToken, self.decoder_embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], self.num_patches, 1],
            )
        )
        x = tf.concat([x[:, :1, :], mask_tokens], axis=1)  # append cls token

        # add pos embed
        x = x + tf.repeat(
            self._cast(self.decoder_pos_embed), repeats=x.shape[0], axis=0
        )

        # apply Transformer blocks
        for j in range(self.decoder_depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                ViTBlock,
                embed_dim=self.decoder_embed_dim,
                nb_heads=self.decoder_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        # predictor projection
        x = self.get(
            "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
        )(x)

        # remove cls token
        x = x[:, 1:, :]
        return x


def mae_factory(
    img_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    reward_pred=True,
    in_chans=3,
    early_conv=False,
):
    encoder = MaskedViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=early_conv,
    )

    decoder = MaskedViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        norm_pix_loss=False,
        reward_pred=reward_pred,
    )
    return encoder, decoder


def flat_vit_factory(
    img_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    in_chans=3,
):
    encoder = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    )
    decoder = ViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    )
    return encoder, decoder
