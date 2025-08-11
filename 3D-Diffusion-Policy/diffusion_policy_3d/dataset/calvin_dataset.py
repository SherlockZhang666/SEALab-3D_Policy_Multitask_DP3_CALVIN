from typing import Dict
import copy
import pickle

import torch
import numpy as np

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.diffuser_actor.dataset_calvin import CalvinDataset as CalvinDatasetBackend


# very hacky, use absolute path

# INSTRUCTION = "/home/tsungwek/repos/3D-Diffusion-Policy/instructions/calvin_task_ABC_D/"
# TRAIN_SET_DIR = "/scratch/packaged_ABC_D/training"
# VAL_SET_DIR = "/scratch/packaged_ABC_D/validation"

INSTRUCTION = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/task_ABC_D_instructions"
TRAIN_SET_DIR = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/packaged_ABC_D_closeloop/training"
VAL_SET_DIR = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/packaged_ABC_D_closeloop/validation"
# INSTRUCTION = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/calvin_debug_dataset_instructions"
# TRAIN_SET_DIR = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/packaged_calvin_debug_dataset_closeloop/training"
# VAL_SET_DIR = "/data/sea_disk0/zhangxx/3d_diffuser_actor/data/calvin/packaged_calvin_debug_dataset_closeloop/validation"
HORIZON = 4
NOBS_STEP = 2
MAX_EPISODE_PER_TASK = -1
CAMERAS = ("front", "wrist")
IMAGE_RESCALE = "0.75,1.25"
RELATIVE_ACTION = True

def load_instructions(instructions, split):
    instructions = pickle.load(
        open(f"{instructions}/{split}.pkl", "rb")
    )['embeddings']
    return instructions


class CalvinDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.pad_before = pad_before
        self.pad_after = pad_after

        train_instruction = load_instructions(
            INSTRUCTION, 'training'
        )
        taskvar = [
            ("A", 0), ("B", 0), ("C", 0), ("D", 0),
        ]
        self.backend_dataset = CalvinDatasetBackend(
            root=TRAIN_SET_DIR,
            instructions=train_instruction,
            taskvar=taskvar,
            horizon=HORIZON,
            nobs_step=NOBS_STEP,
            cache_size=0,
            max_episodes_per_task=MAX_EPISODE_PER_TASK,
            num_iters=None,
            cameras=CAMERAS,
            training=True,
            image_rescale=tuple(
                float(x) for x in IMAGE_RESCALE.split(",")
            ),
            relative_action=RELATIVE_ACTION,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )

    def get_validation_dataset(self):
        test_instruction = load_instructions(
            INSTRUCTION, 'validation'
        )
        taskvar = [
            ("A", 0), ("B", 0), ("C", 0), ("D", 0),
        ]
        self.backend_dataset = CalvinDatasetBackend(
            root=VAL_SET_DIR,
            instructions=test_instruction,
            taskvar=taskvar,
            horizon=HORIZON,
            nobs_step=NOBS_STEP,
            cache_size=0,
            max_episodes_per_task=MAX_EPISODE_PER_TASK,
            num_iters=None,
            cameras=CAMERAS,
            training=False,
            image_rescale=tuple(
                float(x) for x in IMAGE_RESCALE.split(",")
            ),
            relative_action=RELATIVE_ACTION,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        return self

    def get_normalizer(self, mode='limits', **kwargs):
        _horizon = self.backend_dataset._horizon
        _nobs_step = self.backend_dataset._nobs_step

        action, agent_pos, point_cloud = [], [], []
        # for i in range(len(self.backend_dataset)):
        #     for _ in range(15):
        #         ep = self.backend_dataset[i]
        #         action.append(ep['action'])  # (horizon, D_action)
        #         agent_pos.append(ep['agent_pos'])  # (nobs_step, D_pos)
        #         point_cloud.append(ep['pcds'])  # (nobs_step, npts, 3)
        # 减少重复采样，只采样必要的数据量
        sample_size = min(len(self.backend_dataset), 1000)  # 限制采样数量
        sample_indices = np.random.choice(len(self.backend_dataset), sample_size, replace=False)

        for i in sample_indices:
            ep = self.backend_dataset[i]
            if ep is not None:
                action.append(ep['action'])
                agent_pos.append(ep['agent_pos'])
                point_cloud.append(ep['pcds'])
        action = torch.cat(action, dim=0)
        agent_pos = torch.cat(agent_pos, dim=0)
        point_cloud = torch.cat(point_cloud, dim=0)

        action = action.repeat(1, _horizon, 1) # (sample_size, horizon, D_action)
        agent_pos = agent_pos.repeat(1, _nobs_step, 1) # (sample_size, nobs_step, D_pos)
        point_cloud = point_cloud.repeat(1, _nobs_step, 1, 1) # (sample_size, nobs_step, npts, 3)

        data = {
            'action': action,
            'agent_pos': agent_pos,
            'point_cloud': point_cloud,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.backend_dataset)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'].float()
        # We do FPS later
        point_cloud = torch.cat([sample['pcds'], sample['rgbs']], dim=-1).float()
        instr = sample['instr'].float()

        data = {
            'obs': {
                'point_cloud': point_cloud, # nobs, npts, 6
                'agent_pos': agent_pos, # nobs, D_pos
                'instr': instr # n_tokens, D
            },
            'action': sample['action'].float() # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.backend_dataset[idx]
        torch_data = self._sample_to_data(sample)
        return torch_data