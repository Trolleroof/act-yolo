import argparse
import os
import sys
import pickle
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import load_data, set_seed
from imitate_episodes import train_bc

def main():
    parser = argparse.ArgumentParser(description="ACT training script")
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'yolo_guided', 'yolo_crops', 'gt_boxes'])
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--task_name', type=str, default='sim_pick_place')
    # Fair-comparison augmentation. Defaults ON so both modes are trained to
    # cope with the corrupted eval environments (lifts the floor symmetrically).
    parser.add_argument('--image_aug', dest='image_aug', action='store_true', default=True,
                        help='Train-time visual corruption (identical across modes).')
    parser.add_argument('--no_image_aug', dest='image_aug', action='store_false')
    parser.add_argument('--box_aug', dest='box_aug', action='store_true', default=True,
                        help='Jitter/drop detection boxes during training (yolo_guided/gt_boxes).')
    parser.add_argument('--no_box_aug', dest='box_aug', action='store_false')

    args = parser.parse_args()

    set_seed(args.seed)

    ckpt_dir = os.path.join(_PROJECT_ROOT, 'checkpoints', args.mode)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[args.task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Set dimensions
    action_dim = 7
    qpos_dim = 17 if args.mode in ['yolo_guided', 'gt_boxes'] else 7

    policy_config = {
        'lr': args.lr,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'state_dim': action_dim,
        'qpos_dim': qpos_dim,
    }

    config = {
        'num_epochs': args.num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': action_dim,
        'qpos_dim': qpos_dim,
        'lr': args.lr,
        'policy_class': 'ACT',
        'onscreen_render': False,
        'policy_config': policy_config,
        'task_name': args.task_name,
        'seed': args.seed,
        'temporal_agg': args.temporal_agg,
        'camera_names': camera_names,
        'real_robot': False,
        'mode': args.mode
    }

    # Load data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, args.batch_size, args.batch_size,
        mode=args.mode, image_aug=args.image_aug, box_aug=args.box_aug, seed=args.seed
    )
    print(f"Augmentation: image_aug={args.image_aug} box_aug={args.box_aug}")

    # Save stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Train
    best_epoch, min_val_loss, best_state_dict = train_bc(train_dataloader, val_dataloader, config)

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best checkpoint saved to {ckpt_path} with val loss {min_val_loss:.6f} at epoch {best_epoch}')

if __name__ == '__main__':
    main()
