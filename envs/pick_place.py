import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

# qpos layout: [arm(6), left_finger, right_finger, cube_free_joint(7)]
SINGLE_ARM_START_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

DT = 0.02
EPISODE_LEN = 400

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'pick_place.xml')

# Cube spawn range on table (world frame)
CUBE_SPAWN_X = (-0.15, 0.05)
CUBE_SPAWN_Y = (0.50, 0.60)
CUBE_Z = 0.05  # half-height above table surface

# Target zone fixed world position (matches XML)
TARGET_ZONE_POS = np.array([0.1, 0.65, 0.001])
SUCCESS_THRESH = 0.03  # metres

PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

_normalize_gripper = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
_unnormalize_gripper = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE


def make_pick_place_env(random_seed=None):
    physics = mujoco.Physics.from_xml_path(XML_PATH)
    task = PickPlaceTask(random=random_seed)
    env = control.Environment(
        physics, task,
        time_limit=EPISODE_LEN * DT,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class PickPlaceTask(base.Task):
    """Single-arm pick-and-place: grasp red cube, place on green target zone."""

    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1.0

    def initialize_episode(self, physics):
        rng = self.random if self.random is not None else np.random
        with physics.reset_context():
            physics.named.data.qpos[:8] = SINGLE_ARM_START_POSE
            np.copyto(physics.data.ctrl, SINGLE_ARM_START_POSE)
            # Randomize cube XY; orientation upright (quat = 1 0 0 0)
            cx = rng.uniform(*CUBE_SPAWN_X)
            cy = rng.uniform(*CUBE_SPAWN_Y)
            physics.named.data.qpos['cube_joint'] = [cx, cy, CUBE_Z, 1, 0, 0, 0]
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        raw = physics.data.qpos.copy()
        arm = raw[:6]
        gripper = _normalize_gripper(raw[6])
        return np.concatenate([arm, [gripper]])  # (7,)

    @staticmethod
    def get_qvel(physics):
        raw = physics.data.qvel.copy()
        return raw[:7]  # arm (6) + left_finger (1)

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[8:]  # cube free joint (7,)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['top'] = physics.render(height=480, width=480, camera_id='top')
        obs['wrist'] = physics.render(height=480, width=480, camera_id='left_wrist')
        return obs

    def before_step(self, action, physics):
        # action: (7,) = [arm_joints(6), normalized_gripper(1)]
        arm = action[:6]
        gripper = _unnormalize_gripper(np.clip(action[6], 0.0, 1.0))
        # ctrl: [arm(6), left_finger, right_finger]
        ctrl = np.concatenate([arm, [gripper], [-gripper]])
        super().before_step(ctrl, physics)

    def get_reward(self, physics):
        return float(self._is_success(physics))

    @staticmethod
    def _is_success(physics):
        cube_pos = physics.named.data.xpos['cube']
        dist = np.linalg.norm(cube_pos - TARGET_ZONE_POS)
        return dist < SUCCESS_THRESH
