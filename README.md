# Masked World Models for Visual Control

Implementation of the MWM in TensorFlow 2. Our code is based on the implementation of [DreamerV2](https://github.com/danijar/dreamerv2) and [APV](https://arxiv.org/abs/2203.13880). We plan to release the raw data we used for reporting our main experimental results in `data` directory.

## Method
Masked World Models (MWM) is a visual model-based RL algorithm that decouples visual representation learning and dynamics learning. The key idea of MWM is to train an autoencoder that reconstructs visual observations with convolutional feature masking, and a latent dynamics model on top of the autoencoder.

![overview_figure](https://user-images.githubusercontent.com/20944657/176366822-1c755eee-392e-4b7b-a1d0-8a34417888bc.gif)


## Instructions

Get dependencies:
```
pip install tensorflow==2.6.0 tensorflow_text==2.6.0 tensorflow_estimator==2.6.0 tensorflow_probability==0.14.1 ruamel.yaml 'gym[atari]' dm_control tfimm git+https://github.com/rlworkgroup/metaworld.git@a0009ed9a208ff9864a5c1368c04c273bb20dd06#egg=metaworld
```

Below are scripts to reproduce our experimental results. It is possible to run experiments with/without early convolution and reward prediction by leveraging `mae.reward_pred` and `mae.early_conv` arguments.

[Meta-world](https://github.com/rlworkgroup/metaworld) experiments
```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 python mwm/train.py --logdir logs --configs metaworld --task metaworld_peg_insert_side --steps 502000 --mae.reward_pred True --mae.early_conv True
```

[RLBench](https://github.com/stepjam/RLBench) experiments
```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 python mwm/train.py --logdir logs --configs rlbench --task rlbench_reach_target --steps 502000 --mae.reward_pred True --mae.early_conv True
```

[DeepMind Control Suite](https://github.com/deepmind/dm_control) experiments
```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 python mwm/train.py --logdir logs --configs dmc_vision --task dmc_manip_reach_duplo --steps 252000 --mae.reward_pred True --mae.early_conv True
```

## Tips

- Use `TF_XLA_FLAGS=--tf_xla_auto_jit=2 ` to accelerate the training. This requires properly setting your CUDA and CUDNN paths in our machine. You can check this whether `which ptxas` gives you a path to the CUDA/bin path in your machine.

- Also see the tips available in [DreamerV2 repository](https://github.com/danijar/dreamerv2/blob/main/README.md#tips).