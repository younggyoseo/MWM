import atexit
import os
import random
import sys
import threading
import traceback

import cloudpickle
from functools import partial
import gym
import numpy as np


def process_state(s):
    return np.concatenate([s[:1], s[22:-3]])


class GymWrapper:
    def __init__(self, env, obs_key="image", act_key="action"):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._act_is_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs


class DMC:
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ["MUJOCO_GL"] = "egl"
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision")
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite

            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action["action"])
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs


class MetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat

        self._camera = camera

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success += float(info["success"])
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": success,
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
        }
        return obs


class RLBench:
    def __init__(
        self,
        name,
        size=(64, 64),
        action_repeat=1,
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.tasks import ReachTarget

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.front_camera.image_size = size
        obs_config.front_camera.depth = False
        obs_config.front_camera.point_cloud = False
        obs_config.front_camera.mask = False

        action_mode = partial(JointPosition, absolute_mode=False)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=action_mode(), gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=True,
        )
        env.launch()

        if name == "reach_target":
            task = ReachTarget
        else:
            raise ValueError(name)
        self._env = env
        self._task = env.get_task(task)

        _, obs = self._task.reset()
        self._prev_obs = None

        self._size = size
        self._action_repeat = action_repeat

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_shape, dtype=np.float32
        )
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            reward = 0.0
            for i in range(self._action_repeat):
                obs, reward_, terminal = self._task.step(action["action"])
                success, _ = self._task._task.success()
                reward += reward_
                if terminal:
                    break
            self._prev_obs = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            success = False
            reward = 0.0
            obs = self._prev_obs

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
            "image": obs.front_rgb,
            "success": success,
        }
        return obs

    def reset(self):
        _, obs = self._task.reset()
        self._prev_obs = obs
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": obs.front_rgb,
            "success": False,
        }
        return obs


class Dummy:
    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        return {"action": gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
        }

    def reset(self):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:
    def __init__(self, env, key="action"):
        assert hasattr(env.act_space[key], "n")
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
