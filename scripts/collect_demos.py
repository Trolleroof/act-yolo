"""
scripts/collect_demos.py — Scripted pick-and-place demo collector.

Runs a numerical-IK waypoint policy on the vx300s single-arm MuJoCo env,
saves each episode to an HDF5 file, and optionally runs YOLO detection on
the top-camera frames if model weights are present.

Usage:
    python scripts/collect_demos.py --num_episodes 50 --out data/demos

HDF5 layout (per episode):
    top_rgb      (T, 480, 480, 3)  uint8
    wrist_rgb    (T, 480, 480, 3)  uint8
    cube_boxes   (T, 5)            float32  zeros if no YOLO weights
    target_boxes (T, 5)            float32  zeros if no YOLO weights
    qpos         (T, 7)            float32
    actions      (T, 7)            float32
    attrs/success                  bool
"""

import argparse
import os
import sys

# Ensure project root is on the path regardless of where the script is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mujoco
import numpy as np
import h5py

from envs.pick_place import (
    make_pick_place_env,
    PickPlaceTask,
    SINGLE_ARM_START_POSE,
    TARGET_ZONE_POS,
    EPISODE_LEN,
    SUCCESS_THRESH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARM_BASE = np.array([-0.469, 0.5, 0.0])   # robot base in world frame

# Gripper normalised values  (1=open, 0=closed)
GRIPPER_OPEN   = 1.0
GRIPPER_CLOSED = 0.0

# IK hyper-params
IK_ITERS    = 300
IK_LR       = 0.5
IK_TOL      = 0.006   # metres; stop iterating when within this distance

# Phase step budgets (in env steps, DT=0.02 s each)
# Joints have varying kp: waist=800, shoulder=1600, elbow=800, wrist_angle=50.
# Wrist angle (kp=50) drifts significantly under gravity — equilibrium is used
# as a feature rather than a bug: we command a "seed" value and let gravity
# take the arm to a calibrated equilibrium near the cube.
# Total: 40+15+100+80+35+95+35 = 400 ✓
STEPS_HOME_TO_PREGRASP  = 40   # move from home to hover above cube
STEPS_SETTLE            = 15   # hold at pregrasp while cube settles on table
STEPS_PREGRASP_TO_GRASP = 100  # slow descent; allow gravity sag to fully settle
STEPS_CLOSE_GRIPPER     = 80   # close fingers and hold
STEPS_LIFT              = 35   # lift cube
STEPS_TO_ABOVE_TARGET   = 95   # move to above target zone and settle
STEPS_OPEN_GRIPPER      = 35   # open gripper; let cube fall onto target zone
# (no lower-to-target or retract phase — gripper opens at hover height)

# Heights for the static-IK phases (EE = gripper_prop_link body centre).
# Z_GRASP / Z_PLACE: these are IK targets for the "clean" kinematic model.
# The gravity-aware grasp command (see PickPlacePolicy) overrides these for
# the grasp / place phases with calibrated values instead.
Z_PREGRASP   = 0.18   # hover above cube, well clear of table
Z_GRASP      = 0.040  # IK target (unused — overridden by _grasp_cmd)
Z_LIFT       = 0.23   # carry height
Z_ABOVE_TGT  = 0.18   # hover above target zone
Z_PLACE      = 0.040  # IK target (unused — overridden by _place_cmd)

# ---------------------------------------------------------------------------
# Calibrated gravity-aware grasp / place commands
# ---------------------------------------------------------------------------
# Derived by simulating the arm under constant PD control for 300-500 steps
# and measuring equilibrium EE position.  Key dynamics facts:
#   • waist (kp=800, frictionloss=50):  tracks within 0.005 rad
#   • shoulder (kp=1600, frictionloss=60): tracks slowly; converges to within
#     ~0.03 rad of commanded value in 300 steps when gravity is modest
#   • elbow (kp=800, frictionloss=60): large negative equilibrium error when
#     commanded positive (gravity collapse); tracks well when commanded negative
#   • wrist_angle (kp=50): critical — drifts strongly; negative commanded values
#     (e.g. -0.345) are stable; positive values (>1.5) overshoot to ~2.2+
#
# GRASP: command that equilibrates EE near cube center height (~0.040-0.050 m)
# With cmd=[waist_adaptive, 1.04, -0.50, 0, 1.50, 0] the arm equilibrates at
# EE_z ≈ 0.044 m near cube XY when waist is set to waist_atan2 + 0.026.
# The wrist drifts to ~2.2 rad (overshoot from 1.5) which slightly helps.
_GRASP_SHOULDER   = 1.10
_GRASP_ELBOW      = -0.51   # best from scan: eq_EE_z=0.039 (-0.038, 0.565, 0.039)
_GRASP_WRIST      =  1.96   # best from scan: positive overshoot holds wrist up
# Empirical waist offset to compensate for lateral drift under gravity.
# For this (shoulder, elbow, wrist) config, the equilibrium arm drifts
# ~0.066 rad toward smaller waist from the kinematic atan2 direction.
_GRASP_WAIST_BIAS = -0.066  # compensates lateral drift for this (s,e,w) equilibrium

# PLACE: calibrated for target zone (0.100, 0.650).
# cmd=[0.3, 0.728, -0.062, 0, -0.345, 0] equilibrates to EE=(0.096, 0.675, 0.084)
# which is 0.025m XY from target — cube released from z=0.084 falls near target.
_PLACE_WAIST    = 0.300   # fixed — target zone position is fixed
_PLACE_SHOULDER = 0.728
_PLACE_ELBOW    = -0.062
_PLACE_WRIST    = -0.345


# ---------------------------------------------------------------------------
# Numerical IK
# ---------------------------------------------------------------------------

def _numerical_ik(
    phys,
    target_world: np.ndarray,
    q_init: np.ndarray = None,
    n_iter: int = IK_ITERS,
    lr: float = IK_LR,
    tol: float = IK_TOL,
) -> np.ndarray:
    """Gradient-descent IK targeting the gripper_prop_link body.

    This function temporarily perturbs mjdata to compute forward kinematics,
    then fully restores the original simulation state before returning.

    Args:
        phys: dm_control Physics object (used for FK queries via MuJoCo pointers).
        target_world: (3,) desired world-frame EE position.
        q_init: (6,) starting joint angles; defaults to start pose seed.
        n_iter: max gradient-descent iterations.
        lr: step size.
        tol: convergence tolerance in metres.

    Returns:
        (6,) joint angles that place the EE near target_world.
    """
    mjmodel = phys.model.ptr
    mjdata  = phys.data.ptr
    target  = np.asarray(target_world, dtype=float)

    # Save full simulation state so we can restore it after IK.
    saved_qpos = mjdata.qpos.copy()
    saved_qvel = mjdata.qvel.copy()
    saved_ctrl = mjdata.ctrl.copy()
    saved_time = mjdata.time

    if q_init is None:
        q = np.array(SINGLE_ARM_START_POSE[:6], dtype=float)
    else:
        q = np.asarray(q_init, dtype=float).copy()

    jnt_lo = np.array([mjmodel.jnt_range[i][0] for i in range(6)])
    jnt_hi = np.array([mjmodel.jnt_range[i][1] for i in range(6)])

    for _ in range(n_iter):
        # Forward kinematics only (no dynamics, no contact).
        mjdata.qpos[:6] = q
        mujoco.mj_kinematics(mjmodel, mjdata)
        mujoco.mj_comPos(mjmodel, mjdata)
        ee  = phys.named.data.xpos['vx300s_left/gripper_prop_link'].copy()
        err = target - ee

        if np.linalg.norm(err) < tol:
            break

        # Numerical Jacobian (3 × 6)
        J = np.zeros((3, 6))
        for j in range(6):
            dq          = np.zeros(6)
            dq[j]       = 1e-3
            mjdata.qpos[:6] = q + dq
            mujoco.mj_kinematics(mjmodel, mjdata)
            mujoco.mj_comPos(mjmodel, mjdata)
            ee_plus     = phys.named.data.xpos['vx300s_left/gripper_prop_link'].copy()
            J[:, j]     = (ee_plus - ee) / 1e-3

        # Damped pseudo-inverse step.
        dq_vec = lr * np.linalg.pinv(J) @ err
        q      = np.clip(q + dq_vec, jnt_lo, jnt_hi)

    # Restore the full simulation state exactly as it was.
    np.copyto(mjdata.qpos, saved_qpos)
    np.copyto(mjdata.qvel, saved_qvel)
    np.copyto(mjdata.ctrl, saved_ctrl)
    mjdata.time = saved_time
    mujoco.mj_forward(mjmodel, mjdata)

    return q


# ---------------------------------------------------------------------------
# Waypoint policy
# ---------------------------------------------------------------------------

class PickPlacePolicy:
    """Open-loop waypoint policy that adapts to the observed cube position.

    Phase sequence:
        0  HOME → PREGRASP  (open gripper, move above cube)
        1  PREGRASP → GRASP (lower to cube)
        2  CLOSE GRIPPER
        3  LIFT
        4  LIFT → ABOVE TARGET
        5  ABOVE TARGET → PLACE
        6  OPEN GRIPPER
        7  RETRACT (lift up slightly)

    IK is solved once at the start of an episode from the current cube
    position (read via physics.named.data.xpos['cube']).
    """

    def __init__(self, physics, inject_noise: bool = False, noise_scale: float = 0.005):
        self._phys         = physics
        self._inject_noise = inject_noise
        self._noise_scale  = noise_scale

        # Read cube position at episode start.
        cube_pos = physics.named.data.xpos['cube'].copy()

        # Waist angle: points arm base → cube / target in the XY plane.
        def _waist(xy):
            d = np.asarray(xy) - ARM_BASE[:2]
            return float(np.arctan2(d[1], d[0]))

        waist_cube = _waist(cube_pos[:2])
        waist_tgt  = _waist(TARGET_ZONE_POS[:2])

        # ---- IK for pre-grasp and lift (purely static, wrist-up seed) ----
        seed_cube = np.array([waist_cube, 0.30, 0.04, 0.0, 1.1, 0.0])
        seed_tgt  = np.array([waist_tgt,  0.60, -0.30, 0.0, 1.5, 0.0])

        above_cube  = np.array([cube_pos[0], cube_pos[1], Z_PREGRASP])
        lift_pos    = np.array([cube_pos[0], cube_pos[1], Z_LIFT])
        above_tgt   = np.array([TARGET_ZONE_POS[0], TARGET_ZONE_POS[1], Z_ABOVE_TGT])

        q_above   = _numerical_ik(physics, above_cube, q_init=seed_cube)
        q_lift    = _numerical_ik(physics, lift_pos,   q_init=seed_cube)
        q_above_t = _numerical_ik(physics, above_tgt,  q_init=seed_tgt)

        # ---- Gravity-calibrated grasp command ----
        # The waist bias compensates for lateral drift under gravity.
        waist_grasp = np.clip(waist_cube + _GRASP_WAIST_BIAS, -3.14, 3.14)
        q_grasp = np.array([waist_grasp, _GRASP_SHOULDER, _GRASP_ELBOW,
                             0.0, _GRASP_WRIST, 0.0])

        home_q   = np.array(SINGLE_ARM_START_POSE[:6], dtype=float)

        # Waypoints: list of (step_end, q6, gripper)
        # Each waypoint is linearly interpolated from the previous.
        t = 0
        self._waypoints = []

        def add(dt, q6, gripper):
            nonlocal t
            t += dt
            self._waypoints.append((t, q6.copy(), float(gripper)))

        add(STEPS_HOME_TO_PREGRASP,  q_above,   GRIPPER_OPEN)
        add(STEPS_SETTLE,            q_above,   GRIPPER_OPEN)   # hold above while cube settles
        add(STEPS_PREGRASP_TO_GRASP, q_grasp,   GRIPPER_OPEN)
        add(STEPS_CLOSE_GRIPPER,     q_grasp,   GRIPPER_CLOSED)
        add(STEPS_LIFT,              q_lift,    GRIPPER_CLOSED)
        add(STEPS_TO_ABOVE_TARGET,   q_above_t, GRIPPER_CLOSED)
        add(STEPS_OPEN_GRIPPER,      q_above_t, GRIPPER_OPEN)   # release cube at hover height

        # Prepend a "time-0" entry at the home pose.
        self._waypoints.insert(0, (0, home_q, GRIPPER_OPEN))
        self._step = 0

    def __call__(self) -> np.ndarray:
        """Return a 7-DOF action for the current step."""
        t = self._step

        # Find surrounding waypoints.
        wp = self._waypoints
        idx = 0
        while idx + 1 < len(wp) - 1 and wp[idx + 1][0] <= t:
            idx += 1

        t0, q0, g0 = wp[idx]
        t1, q1, g1 = wp[idx + 1]

        if t1 > t0:
            frac = float(t - t0) / float(t1 - t0)
            frac = np.clip(frac, 0.0, 1.0)
        else:
            frac = 1.0

        q_interp = q0 + frac * (q1 - q0)
        g_interp = g0 + frac * (g1 - g0)
        g_interp = float(np.clip(g_interp, 0.0, 1.0))

        if self._inject_noise:
            q_interp = q_interp + np.random.uniform(
                -self._noise_scale, self._noise_scale, q_interp.shape
            )

        action = np.concatenate([q_interp, [g_interp]])
        self._step += 1
        return action

    @property
    def done(self) -> bool:
        last_t = self._waypoints[-1][0]
        return self._step >= last_t


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def make_renderers(mjmodel):
    """Return (top_renderer, wrist_renderer) using native MuJoCo renderer."""
    top   = mujoco.Renderer(mjmodel, 480, 480)
    wrist = mujoco.Renderer(mjmodel, 480, 480)
    return top, wrist


def render_cameras(top_r, wrist_r, mjdata):
    top_r.update_scene(mjdata, camera='top')
    top_frame   = top_r.render().copy()

    wrist_r.update_scene(mjdata, camera='left_wrist')
    wrist_frame = wrist_r.render().copy()

    return top_frame, wrist_frame


# ---------------------------------------------------------------------------
# YOLO helper (optional)
# ---------------------------------------------------------------------------

def _try_load_yolo(weights_path: str):
    """Load YOLODetector if weights file exists, else return None."""
    if not os.path.isfile(weights_path):
        return None
    try:
        sys.path.insert(0, _PROJECT_ROOT)
        from detection.yolo_detector import YOLODetector
        detector = YOLODetector(weights=weights_path)
        print(f"[YOLO] Loaded weights from {weights_path}")
        return detector
    except Exception as exc:
        print(f"[YOLO] Could not load detector: {exc}  — storing zeros instead.")
        return None


def _detect_boxes(detector, top_rgb: np.ndarray):
    """Run YOLO; return (cube_box, target_box) each (5,) float32 or zeros."""
    zero = np.zeros(5, dtype=np.float32)
    if detector is None:
        return zero.copy(), zero.copy()
    try:
        dets = detector.detect(top_rgb)
        cube_box   = np.array(dets['cube'],        dtype=np.float32) if dets['cube']        is not None else zero.copy()
        target_box = np.array(dets['target_zone'], dtype=np.float32) if dets['target_zone'] is not None else zero.copy()
    except Exception:
        cube_box, target_box = zero.copy(), zero.copy()
    return cube_box, target_box


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, episode_idx: int, detector, inject_noise: bool = False):
    """Roll out one episode and return a dict of arrays.

    Returns:
        dict with keys: top_rgb, wrist_rgb, cube_boxes, target_boxes,
                        qpos, actions, success
    """
    ts = env.reset()
    phys = env._physics

    # Build native MuJoCo renderer (shares model/data with dm_control physics).
    mjmodel = phys.model.ptr
    mjdata  = phys.data.ptr
    top_r, wrist_r = make_renderers(mjmodel)

    # Instantiate policy (IK solved here, after reset so cube pos is set).
    policy = PickPlacePolicy(phys, inject_noise=inject_noise)

    # Storage lists
    top_frames   = []
    wrist_frames = []
    cube_boxes   = []
    target_boxes = []
    qpos_list    = []
    action_list  = []

    success = False
    for step_i in range(EPISODE_LEN):
        # Render with native MuJoCo renderer.
        top_frame, wrist_frame = render_cameras(top_r, wrist_r, mjdata)

        # YOLO detection on top frame (or zeros).
        cb, tb = _detect_boxes(detector, top_frame)

        # Current qpos from dm_control observation.
        q_obs = PickPlaceTask.get_qpos(phys).astype(np.float32)  # (7,)

        # Policy action.
        action = policy().astype(np.float32)  # (7,)

        # Store.
        top_frames.append(top_frame)
        wrist_frames.append(wrist_frame)
        cube_boxes.append(cb)
        target_boxes.append(tb)
        qpos_list.append(q_obs)
        action_list.append(action)

        # Step environment with dm_control.
        ts = env.step(action)

        # Check success every step.
        if ts.reward > 0:
            success = True

    # Close renderers to free GPU/GPU resources.
    top_r.close()
    wrist_r.close()

    return {
        'top_rgb':      np.stack(top_frames,   axis=0).astype(np.uint8),
        'wrist_rgb':    np.stack(wrist_frames, axis=0).astype(np.uint8),
        'cube_boxes':   np.stack(cube_boxes,   axis=0).astype(np.float32),
        'target_boxes': np.stack(target_boxes, axis=0).astype(np.float32),
        'qpos':         np.stack(qpos_list,    axis=0).astype(np.float32),
        'actions':      np.stack(action_list,  axis=0).astype(np.float32),
        'success':      bool(success),
    }


# ---------------------------------------------------------------------------
# HDF5 saving
# ---------------------------------------------------------------------------

def save_episode(out_dir: str, episode_idx: int, data: dict):
    """Write episode data to data/demos/episode_{N}.hdf5."""
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f'episode_{episode_idx}.hdf5')

    with h5py.File(fpath, 'w') as f:
        f.create_dataset('top_rgb',      data=data['top_rgb'],      compression='gzip', compression_opts=4)
        f.create_dataset('wrist_rgb',    data=data['wrist_rgb'],    compression='gzip', compression_opts=4)
        f.create_dataset('cube_boxes',   data=data['cube_boxes'])
        f.create_dataset('target_boxes', data=data['target_boxes'])
        f.create_dataset('qpos',         data=data['qpos'])
        f.create_dataset('actions',      data=data['actions'])
        f.attrs['success'] = data['success']

    return fpath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Collect scripted pick-and-place demos.')
    p.add_argument('--num_episodes', type=int,  default=50,
                   help='Total episodes to attempt (default: 50).')
    p.add_argument('--out',          type=str,  default='data/demos',
                   help='Output directory for HDF5 files (default: data/demos).')
    p.add_argument('--seed',         type=int,  default=0,
                   help='Base random seed (incremented per episode).')
    p.add_argument('--inject_noise', action='store_true',
                   help='Add small Gaussian noise to actions for diversity.')
    p.add_argument('--yolo_weights', type=str,
                   default='weights/yolov8n_pickplace.pt',
                   help='Path to YOLO weights (relative to project root).')
    p.add_argument('--smoke_test',   action='store_true',
                   help='Run 2 episodes and print HDF5 shapes then exit.')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output directory relative to project root if not absolute.
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(_PROJECT_ROOT, args.out)

    # Optionally load YOLO.
    yolo_weights_abs = (
        args.yolo_weights if os.path.isabs(args.yolo_weights)
        else os.path.join(_PROJECT_ROOT, args.yolo_weights)
    )
    detector = _try_load_yolo(yolo_weights_abs)
    if detector is None:
        print('[YOLO] No weights found — cube_boxes / target_boxes will be zeros.')

    num_episodes = 2 if args.smoke_test else args.num_episodes
    successes    = 0

    # Create env once and reuse (reset() is called inside run_episode).
    env = make_pick_place_env(random_seed=args.seed)

    last_saved_path = None
    for ep_idx in range(num_episodes):
        # Reset with a different seed each episode for randomised cube placement.
        env._task._random = np.random.RandomState(args.seed + ep_idx)

        print(f'Episode {ep_idx + 1}/{num_episodes} ... ', end='', flush=True)
        data = run_episode(env, ep_idx, detector, inject_noise=args.inject_noise)

        fpath = save_episode(out_dir, ep_idx, data)
        last_saved_path = fpath

        status = 'SUCCESS' if data['success'] else 'fail'
        if data['success']:
            successes += 1

        print(f'{status}  →  {fpath}')

    print(f'\n=== Summary: {successes} / {num_episodes} successful ===')

    # Smoke-test: print shapes from the last saved HDF5.
    if args.smoke_test and last_saved_path:
        print('\n--- Smoke-test: HDF5 shapes ---')
        with h5py.File(last_saved_path, 'r') as f:
            for key in ['top_rgb', 'wrist_rgb', 'cube_boxes', 'target_boxes', 'qpos', 'actions']:
                ds = f[key]
                print(f'  {key:20s}: shape={ds.shape}  dtype={ds.dtype}')
            print(f'  {"success":20s}: {f.attrs["success"]}')


if __name__ == '__main__':
    main()
