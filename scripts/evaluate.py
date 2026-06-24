import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.pick_place import make_pick_place_env
from vision.corruption import corrupt_obs_images
from policy import ACTPolicy
from utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

def get_gt_bboxes(physics):
    import mujoco
    from detection.generate_yolo_data import get_bbox_from_mask
    
    cube_geom_id = physics.model.name2id('cube_geom', 'geom')
    target_geom_id = physics.model.name2id('target_zone_geom', 'geom')
    
    renderer_seg = mujoco.Renderer(physics.model.ptr, height=480, width=480)
    renderer_seg.enable_segmentation_rendering()
    renderer_seg.update_scene(physics.data.ptr, camera='top')
    seg = renderer_seg.render()
    geom_ids = seg[:, :, 0]
    renderer_seg.close()
    
    cube_mask = (geom_ids == cube_geom_id).astype(np.uint8)
    target_mask = (geom_ids == target_geom_id).astype(np.uint8)
    
    cube_bbox = get_bbox_from_mask(cube_mask)
    target_bbox = get_bbox_from_mask(target_mask)
    
    cube_box = np.array(list(cube_bbox) + [1.0], dtype=np.float32) if cube_bbox is not None else np.zeros(5, dtype=np.float32)
    target_box = np.array(list(target_bbox) + [1.0], dtype=np.float32) if target_bbox is not None else np.zeros(5, dtype=np.float32)
    
    return cube_box, target_box

def get_image_from_obs(obs, camera_names):
    from einops import rearrange
    curr_images = []
    for cam_name in camera_names:
        frame = obs[cam_name]
        curr_image = rearrange(frame, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def annotate_frame(frame, dets, success=None):
    annotated = frame.copy()
    if dets is not None:
        for name, box in dets.items():
            if box is not None:
                cx, cy, w, h, conf = box
                H, W = annotated.shape[:2]
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                
                color = (0, 0, 255) if name == 'cube' else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{name}: {conf:.2f}", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
    if success is not None:
        color = (0, 255, 0) if success else (0, 0, 255)
        text = "SUCCESS" if success else "FAILED"
        cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
    return annotated

def main():
    parser = argparse.ArgumentParser(description="ACT evaluation under visual corruption")
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'yolo_guided', 'yolo_crops', 'gt_boxes'])
    parser.add_argument('--corruption_severity', type=str, default='0',
                        help="0/clean, 1/low, 2/medium, 3/high")
    parser.add_argument('--num_rollouts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--task_name', type=str, default='sim_pick_place')
    
    args = parser.parse_args()

    # Parse severity
    try:
        severity = int(args.corruption_severity)
    except ValueError:
        aliases = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3, 'light': 1, 'heavy': 3}
        severity = aliases[args.corruption_severity.lower()]

    set_seed(args.seed)

    ckpt_dir = os.path.join(_PROJECT_ROOT, 'checkpoints', args.mode)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')

    if not os.path.isfile(ckpt_path):
        print(f"Warning: Best checkpoint not found at {ckpt_path}. Falling back to last checkpoint.")
        ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')

    if not os.path.isfile(ckpt_path):
        print(f"Error: No policy found at {ckpt_dir}")
        sys.exit(1)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Resolve dimensions
    action_dim = 7
    qpos_dim = 17 if args.mode in ['yolo_guided', 'gt_boxes'] else 7
    camera_names = ['top', 'wrist']
    episode_len = 400

    policy_config = {
        'lr': 1e-5,
        'num_queries': 100, # chunk size
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'state_dim': action_dim,
        'qpos_dim': qpos_dim,
    }

    # Load policy
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.to(device)
    policy.eval()

    # Load YOLO if needed
    detector = None
    if args.mode in ['yolo_guided', 'yolo_crops']:
        yolo_path = os.path.join(_PROJECT_ROOT, 'weights', 'yolov8n_pickplace.pt')
        from detection.yolo_detector import YOLODetector
        detector = YOLODetector(weights=yolo_path, conf=0.4)

    # Setup normalization
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    env = make_pick_place_env(random_seed=args.seed)

    success_count = 0
    total_steps = 0
    cube_misses = 0
    target_misses = 0

    print(f"Evaluating {args.mode} on task {args.task_name} (Severity {severity}) for {args.num_rollouts} rollouts...")

    # For saving annotated videos
    os.makedirs(os.path.join(_PROJECT_ROOT, 'data', 'eval_videos'), exist_ok=True)
    video_path = os.path.join(_PROJECT_ROOT, 'data', 'eval_videos', f"{args.mode}_sev{severity}.mp4")
    video_writer = None

    for rollout_id in tqdm(range(args.num_rollouts)):
        # Make episode layout deterministic across different model evals
        env._task._random = np.random.RandomState(args.seed + rollout_id)
        ts = env.reset()
        
        # Setup temporal aggregation
        num_queries = 100
        all_time_actions = torch.zeros([episode_len, episode_len + num_queries, action_dim]).to(device)
        
        success = False
        rollout_frames = []

        with torch.inference_mode():
            for t in range(episode_len):
                # Apply visual corruption to observations
                obs = dict(ts.observation)
                obs = corrupt_obs_images(obs, severity)

                # Process observations based on mode
                dets = None
                if args.mode == 'yolo_guided':
                    dets = detector.detect(obs['top'])
                    cube_box = dets['cube'] if dets['cube'] is not None else np.zeros(5)
                    target_box = dets['target_zone'] if dets['target_zone'] is not None else np.zeros(5)
                    
                    total_steps += 1
                    if dets['cube'] is None: cube_misses += 1
                    if dets['target_zone'] is None: target_misses += 1
                    
                    qpos_numpy = np.concatenate([obs['qpos'], cube_box, target_box])
                elif args.mode == 'gt_boxes':
                    cube_box, target_box = get_gt_bboxes(env._physics)
                    qpos_numpy = np.concatenate([obs['qpos'], cube_box, target_box])
                elif args.mode == 'yolo_crops':
                    dets = detector.detect(obs['top'])
                    
                    total_steps += 1
                    if dets['cube'] is None: cube_misses += 1
                    if dets['target_zone'] is None: target_misses += 1
                    
                    obs['top'] = detector.get_crop(obs['top'], 'cube', dets=dets)
                    qpos_numpy = obs['qpos']
                else:  # baseline
                    qpos_numpy = obs['qpos']

                # Normalize and package qpos
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

                # Package camera images
                curr_image = get_image_from_obs(obs, camera_names)

                # Query ACT policy
                if t % 1 == 0:  # query frequency is 1 when temporal_agg is active
                    all_actions = policy(qpos, curr_image)
                
                # Temporal aggregation weighting
                all_time_actions[[t], t:t+num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                # Post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)

                # Step environment
                ts = env.step(action)
                if ts.reward > 0:
                    success = True

                # Record frames for the video from the first rollout
                if rollout_id == 0:
                    # Draw bboxes on the frame
                    annotated_top = annotate_frame(obs['top'], dets)
                    # Convert to BGR for cv2
                    top_bgr = cv2.cvtColor(annotated_top, cv2.COLOR_RGB2BGR)
                    rollout_frames.append(top_bgr)

        if success:
            success_count += 1

        # Write video for rollout 0
        if rollout_id == 0 and len(rollout_frames) > 0:
            h, w = rollout_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            for frame in rollout_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"Saved annotated rollout video for Rollout 0 to {video_path}")

    # Compute metrics
    success_rate = success_count / args.num_rollouts
    cube_miss_rate = (cube_misses / total_steps) if total_steps > 0 else 0.0
    target_miss_rate = (target_misses / total_steps) if total_steps > 0 else 0.0
    overall_miss_rate = ((cube_misses + target_misses) / (2 * total_steps)) if total_steps > 0 else 0.0

    print(f"Results for {args.mode} (Severity {severity}):")
    print(f"  Success Rate:     {success_rate:.3f} ({success_count}/{args.num_rollouts})")
    if total_steps > 0:
        print(f"  Cube Miss Rate:   {cube_miss_rate:.3f}")
        print(f"  Target Miss Rate: {target_miss_rate:.3f}")

    # Save results to JSON
    result_data = {
        'mode': args.mode,
        'severity': severity,
        'num_rollouts': args.num_rollouts,
        'success_rate': success_rate,
        'cube_miss_rate': cube_miss_rate,
        'target_miss_rate': target_miss_rate,
        'overall_miss_rate': overall_miss_rate
    }
    
    os.makedirs(os.path.join(_PROJECT_ROOT, 'data', 'eval_results'), exist_ok=True)
    result_path = os.path.join(_PROJECT_ROOT, 'data', 'eval_results', f"results_{args.mode}_{severity}.json")
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Results JSON saved to {result_path}")

if __name__ == '__main__':
    main()
