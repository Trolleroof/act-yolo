import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

try:
    import IPython
    e = IPython.embed
except ImportError:
    e = lambda: None

# Eval-matched corruption operator, reused for train-time domain randomization.
from vision.corruption import corrupt_frame


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, mode='baseline',
                 image_aug=False, box_aug=False, aug_severities=(0, 1, 2, 3),
                 aug_p=0.85, box_jitter=0.02, box_dropout=0.1, seed=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.mode = mode
        # --- Fair-comparison augmentation (training split only) ---
        # image_aug: corrupt camera frames at train time with the SAME operator
        #   used at eval, applied identically for every mode so the only thing
        #   that differs between baseline and yolo_guided is the box channel.
        # box_aug: jitter / occasionally drop the appended detection boxes so the
        #   policy learns to tolerate the imperfect boxes a real YOLO produces on
        #   corrupted frames (closes the clean-train / noisy-eval box gap).
        self.image_aug = image_aug
        self.box_aug = box_aug
        self.aug_severities = np.asarray(aug_severities)
        self.aug_p = aug_p
        self.box_jitter = box_jitter
        self.box_dropout = box_dropout
        self._rng = np.random.default_rng(seed)
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def _augment_box(self, box):
        """Jitter a (5,) [cx,cy,w,h,conf] box, or zero it to mimic a missed detection."""
        if box is None or not np.any(box):
            return box
        if self._rng.random() < self.box_dropout:
            return np.zeros_like(box)
        out = box.astype(np.float32).copy()
        out[:4] += self._rng.normal(0.0, self.box_jitter, size=4)
        out[:4] = np.clip(out[:4], 0.0, 1.0)
        return out

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs.get('sim', True)
            original_action_shape = root['actions'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            
            # Read qpos
            qpos = root['qpos'][start_ts]
            
            # Concatenate bbox inputs depending on mode
            if self.mode == 'yolo_guided':
                cube_box = root['cube_boxes'][start_ts]
                target_box = root['target_boxes'][start_ts]
                if self.box_aug:
                    cube_box = self._augment_box(cube_box)
                    target_box = self._augment_box(target_box)
                qpos = np.concatenate([qpos, cube_box, target_box])
            elif self.mode == 'gt_boxes':
                gt_cube_box = root['gt_cube_boxes'][start_ts]
                gt_target_box = root['gt_target_boxes'][start_ts]
                if self.box_aug:
                    gt_cube_box = self._augment_box(gt_cube_box)
                    gt_target_box = self._augment_box(gt_target_box)
                qpos = np.concatenate([qpos, gt_cube_box, gt_target_box])

            # Read camera frames
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name == 'top':
                    if self.mode == 'yolo_crops' and 'top_crops' in root:
                        image_dict[cam_name] = root['top_crops'][start_ts]
                    else:
                        image_dict[cam_name] = root['top_rgb'][start_ts]
                elif cam_name == 'wrist':
                    image_dict[cam_name] = root['wrist_rgb'][start_ts]
                else:
                    image_dict[cam_name] = root[cam_name][start_ts]

            # Train-time visual domain randomization (identical across modes).
            # Crops in yolo_crops mode are left untouched (already object-centered).
            if self.image_aug:
                for cam_name in self.camera_names:
                    if self.mode == 'yolo_crops' and cam_name == 'top':
                        continue
                    if self._rng.random() < self.aug_p:
                        sev = int(self._rng.choice(self.aug_severities))
                        if sev > 0:
                            image_dict[cam_name] = corrupt_frame(
                                image_dict[cam_name], sev, rng=self._rng)

            action = root['actions'][start_ts:]
            action_len = episode_len - start_ts

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, episode_ids, mode='baseline'):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in episode_ids:
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['qpos'][()]
            if mode == 'yolo_guided':
                cube_boxes = root['cube_boxes'][()]
                target_boxes = root['target_boxes'][()]
                qpos = np.concatenate([qpos, cube_boxes, target_boxes], axis=-1)
            elif mode == 'gt_boxes':
                gt_cube_boxes = root['gt_cube_boxes'][()]
                gt_target_boxes = root['gt_target_boxes'][()]
                qpos = np.concatenate([qpos, gt_cube_boxes, gt_target_boxes], axis=-1)
            action = root['actions'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def get_successful_episode_ids(dataset_dir):
    """Return sorted IDs of episodes where attrs/success is True."""
    import glob
    ids = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))):
        with h5py.File(path, 'r') as f:
            if f.attrs.get('success', True):
                ids.append(int(os.path.basename(path).split('_')[1].split('.')[0]))
    return ids


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
              mode='baseline', image_aug=False, box_aug=False, seed=None):
    print(f'\nData from: {dataset_dir}\n')
    all_ids = get_successful_episode_ids(dataset_dir)
    if not all_ids:
        raise ValueError(f'No successful episodes found in {dataset_dir}')
    print(f'Using {len(all_ids)} successful episodes')

    train_ratio = 0.8
    perm = np.random.permutation(len(all_ids))
    shuffled_ids = np.array(all_ids)[perm]
    train_indices = shuffled_ids[:int(train_ratio * len(all_ids))]
    val_indices = shuffled_ids[int(train_ratio * len(all_ids)):]

    norm_stats = get_norm_stats(dataset_dir, all_ids, mode)

    # Augmentation is applied to the training split only; validation stays clean
    # so val loss tracks fit to the (clean) demonstrations consistently.
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, mode,
                                    image_aug=image_aug, box_aug=box_aug, seed=seed)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, mode)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
