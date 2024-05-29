from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path

import torch
import einops

from .dataset_engine import RLBenchDataset
from .utils import Resize, TrajectoryInterpolator
from third_party.diffuser_actor.utils_with_calvin import to_relative_action, convert_rotation


class CalvinDataset(RLBenchDataset):

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        horizon=4,
        nobs_step=2,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        relative_action=True,
        pad_before=0,
        pad_after=0,
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = horizon + nobs_step
        self._horizon = horizon
        self._nobs_step = nobs_step
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action
        self._pad_before = pad_before
        self._pad_after = pad_after

        # Keep variations and useful instructions
        self._instructions = instructions
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per-task and variation
        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            self._episodes += eps
            self._num_episodes += len(eps)

        print(f"Created dataset from {root} with {self._num_episodes}")

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (nframe, n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [gripper_tensors],  # wrt frame_ids, (nframe, 8)
            [camera_dicts],
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        total_frame_ids = episode[0]
        total_frame_ids = (total_frame_ids[:1] * self._pad_before +
                           total_frame_ids +
                           total_frame_ids[-1:] * self._pad_after)
        chunk = random.randint(
            0, len(total_frame_ids) - self._max_episode_length - 1
        )

        # Get frame ids for this chunk
        frame_ids = total_frame_ids[chunk : chunk+self._max_episode_length]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        # Sample one instruction feature
        if self._instructions is not None:
            instr_ind = episode[4][0]
            instr = torch.as_tensor(self._instructions[instr_ind])[0]
        else:
            instr = torch.zeros((53, 512))

        # Get gripper tensors for respective frame ids
        gripper = episode[2][frame_ids]
        agent_pose = gripper[:self._nobs_step]
        action = gripper[self._nobs_step:]
        rgbs = rgbs[:self._nobs_step].flatten(1, 2)
        pcds = pcds[:self._nobs_step].flatten(1, 2)

        # Compute relative action
        if self._relative_action:
            rel_action = torch.zeros_like(action)
            for i in range(action.shape[0]):
                rel_action[i] = torch.as_tensor(to_relative_action(
                    action[i].numpy(), agent_pose[-1].numpy(), clip=False
                ))
            action = rel_action

        # Convert Euler angles to Quarternion
        action = torch.cat([
            action[..., :3],
            torch.as_tensor(convert_rotation(action[..., 3:6])),
            action[..., 6:]
        ], dim=-1)
        agent_pose = torch.cat([
            agent_pose[..., :3],
            torch.as_tensor(convert_rotation(agent_pose[..., 3:6])),
            agent_pose[..., 6:]
        ], dim=-1)


        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, npts, 3)
            "pcds": pcds,  # e.g. tensor (n_frames, npts, 3)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "agent_pos": agent_pose,
        }

        return ret_dict
