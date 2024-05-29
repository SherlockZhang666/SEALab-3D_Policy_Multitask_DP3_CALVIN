from typing import Optional
import logging

import transformers
import torch
import torch.nn as nn
import numpy as np
import einops
import dgl.geometry as dgl_geo

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from diffusion_policy_3d.common.pytorch_util import dict_apply
from third_party.diffuser_actor.online_evaluation_calvin.evaluate_utils import convert_action
from third_party.diffuser_actor.utils_with_calvin import relative_to_absolute


logger = logging.getLogger(__name__)


def create_model(args, policy, policy_cfg):
    model = DiffusionModel(args, policy, policy_cfg)

    return model


class DiffusionModel(CalvinBaseModel):
    """A wrapper for the DiffuserActor model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
            4. Action post-processing
                - quaternion to Euler angles
                - relative to absolute action
    """
    def __init__(self, args, policy, policy_cfg):
        self.args = args
        self.policy = policy
        self.policy_cfg = policy_cfg 
        self.text_tokenizer, self.text_model = self.get_text_encoder()
        self.reset()

    def get_text_encoder(self):
        def load_model(encoder) -> transformers.PreTrainedModel:
            if encoder == "bert":
                model = transformers.BertModel.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(model, transformers.PreTrainedModel):
                raise ValueError(f"Unexpected encoder {encoder}")
            return model


        def load_tokenizer(encoder) -> transformers.PreTrainedTokenizer:
            if encoder == "bert":
                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                tokenizer = transformers.CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
                raise ValueError(f"Unexpected encoder {encoder}")
            return tokenizer


        tokenizer = load_tokenizer(self.args.text_encoder)
        tokenizer.model_max_length = self.args.text_max_length

        model = load_model(self.args.text_encoder)
    
        return tokenizer, model

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.args.device
        self.policy.eval()
        self.text_model.eval()

        self.policy = self.policy.to(device)
        self.text_model = self.text_model.to(device)

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        instr = instruction + '.'
        tokens = self.text_tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state

        return pred

    def step(self, obs, instruction):
        """
        Args:
            obs: a dictionary of observations
                - rgb_obs: a dictionary of RGB images
                - depth_obs: a dictionary of depth images
                - robot_obs: a dictionary of proprioceptive states
            lang_annotation: a string indicates the instruction of the task

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Organize inputs
        rgbs = np.stack([
            obs["dp3_obs"]["rgb_obs"]["rgb_static"],
            obs["dp3_obs"]["rgb_obs"]["rgb_gripper"]
        ], axis=1) # [T, ncam, H, W, 3]
        pcds = np.stack([
            obs["dp3_obs"]["pcd_obs"]["pcd_static"],
            obs["dp3_obs"]["pcd_obs"]["pcd_gripper"]
        ], axis=1) # [T, ncam, H, W, 3]

        rgbs = torch.as_tensor(rgbs).to(device)
        pcds = torch.as_tensor(pcds).to(device)

        # Crop the images.  See Line 165-166 in datasets/dataset_calvin.py
        nobs_steps, ncam = rgbs.shape[:2]
        rgbs = rgbs[:, :, 20:180, 20:180, :]
        pcds = pcds[:, :, 20:180, 20:180, :]

        rgbs = einops.rearrange(rgbs, "T N H W C -> (T N) (H W) C")
        pcds = einops.rearrange(pcds, "T N H W C -> (T N) (H W) C")

        # Run FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            pcds.cuda().to(torch.float64), self.args.fps_num_pts, 0
        ).long()
        pcds = torch.gather(
            pcds, 1, sampled_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        rgbs = torch.gather(
            rgbs, 1, sampled_inds.unsqueeze(-1).expand(-1, -1, 3)
        )
        point_cloud = torch.cat([pcds, rgbs], dim=-1)
        point_cloud = einops.rearrange(
            point_cloud, "(T N) P C -> T (N P) C", T=nobs_steps, N=ncam
        )
        point_cloud = point_cloud.unsqueeze(0)

        # history of actions
        gripper = torch.as_tensor(obs["proprio"]).to(device).unsqueeze(0)

        # create obs dict
        obs_dict = {
            'point_cloud': point_cloud.float(), # nobs, npts, 6
            'agent_pos': gripper.float(), # nobs, D_pos
            'instr': instruction.float()
        }

        # run policy
        with torch.no_grad():
            trajectory = self.policy.predict_action(obs_dict)['action']

        # Convert quaternion to Euler angles
        trajectory = convert_action(trajectory)

        if bool(self.args.relative_action):
            # Convert quaternion to Euler angles
            gripper = convert_action(gripper[:, [-1], :])
            # Convert relative action to absolute action
            trajectory = relative_to_absolute(trajectory, gripper)

        # Bound final action by CALVIN statistics
        if self.args.calvin_gripper_loc_bounds is not None:
            trajectory[:, :, :3] = np.clip(
                trajectory[:, :, :3],
                a_min=self.args.calvin_gripper_loc_bounds[0].reshape(1, 1, 3),
                a_max=self.args.calvin_gripper_loc_bounds[1].reshape(1, 1, 3)
            )

        return trajectory
