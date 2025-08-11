"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import os
import gc
from typing import Tuple, Optional, List
import random
import logging
from pathlib import Path

# import tap
import argparse
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml
from tqdm import tqdm

from third_party.diffuser_actor.common_utils import get_gripper_loc_bounds
from third_party.diffuser_actor.online_evaluation_calvin.evaluate_model import create_model
from third_party.diffuser_actor.online_evaluation_calvin.evaluate_utils import (
    prepare_visual_states,
    prepare_proprio_states,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from third_party.diffuser_actor.online_evaluation_calvin.multistep_sequences import get_sequences
from third_party.diffuser_actor.online_evaluation_calvin.evaluate_utils import get_env
from train import TrainDP3Workspace

logger = logging.getLogger(__name__)

EP_LEN = 120  # 120 would be better
NUM_SEQUENCES = 50


# class Arguments(tap.Tap):
#     # Online enviornment
#     # calvin_dataset_path: Path = "/home/tsungwek/repos/calvin/dataset/task_ABC_D"
#     calvin_dataset_path: Path = "/home/lxb/Public/zhx/calvin/dataset/calvin_debug_dataset"
#     # calvin_model_path: Path = "/home/tsungwek/repos/calvin/calvin_models"
#     calvin_model_path: Path = "/home/lxb/Public/zhx/calvin/calvin_models"
#     calvin_demo_tasks: Optional[List[str]] = None
#     device: str = "cuda"
#     text_encoder: str = "clip"
#     text_max_length: int = 16
#     save_video: int = 0

#     # Offline data loader
#     seed: int = 0
#     output_dir: Path
#     calvin_gripper_loc_bounds: Optional[str] = None
#     config_name: str = "instr_dp3.yaml"

#     # Logging to base_log_dir/exp_log_dir/run_log_dir
#     base_log_dir: Path = Path(__file__).parent / "eval_logs" / "calvin"
#     relative_action: int = 1

#     # Model
#     fps_num_pts: int = 1024

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Policy")

    # Online environment
    parser.add_argument("--calvin_dataset_path", type=Path, default="/data/sea_disk0/zhangxx/calvin/dataset/task_ABC_D",
                       help="Path to CALVIN dataset")
    parser.add_argument("--calvin_model_path", type=Path, default="/data/sea_disk0/zhangxx/3DDA_DP3/calvin/calvin_models",
                       help="Path to CALVIN models")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--text_encoder", type=str, default="clip",
                       help="Text encoder to use")
    parser.add_argument("--text_max_length", type=int, default=16,
                       help="Maximum length of text input")
    parser.add_argument("--save_video", type=int, default=0,
                       help="Whether to save video (1/0)")

    # Offline data loader
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--calvin_gripper_loc_bounds", type=str, default=None,
                       help="Path to gripper bounds config")
    parser.add_argument("--config_name", type=str, default="instr_dp3.yaml",
                       help="Config file name")

    # Logging
    parser.add_argument("--base_log_dir", type=Path, default=Path(__file__).parent / "eval_logs" / "ABC_D_closeloop_3000steps_160image",
                       help="Base log directory")
    parser.add_argument("--relative_action", type=int, default=1,
                       help="Use relative actions (1/0)")

    # Model
    parser.add_argument("--fps_num_pts", type=int, default=1024,
                       help="Number of points for FPS")

    args = parser.parse_args()
    
    # Handle local_rank for DDP
    # args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return args


def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env


def evaluate_policy(model, env, conf_dir, eval_log_dir=None, save_video=False,
                    sequence_indices=[]):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        conf_dir: Path to the directory containing the config files of CALVIN
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    # Create instructions.txt file to store task instructions separately from results
    instructions_txt_path = eval_log_dir / "instructions.txt"
    with open(instructions_txt_path, "w") as f:
        f.write("Sequence task instructions:\n\n")

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        
        # Write instructions to instructions.txt
        with open(instructions_txt_path, "a") as f:
            f.write(f"Sequence {seq_ind} tasks:\n")
            for task_idx, subtask in enumerate(eval_sequence):
                lang_annotation = val_annotations[subtask][0]
                f.write(f"  Task {task_idx+1}: {lang_annotation}\n")
            f.write("\n")

        result, videos = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video
        )
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip
            # from moviepy.editor import vfx
            clip = []
            import cv2
            for task_ind, (subtask, video) in enumerate(zip(eval_sequence, videos)):
                for img_ind, img in enumerate(video):
                    cv2.putText(img.copy(),
                                f'{task_ind}: {subtask}',
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (0, 0, 0),
                                1,
                                2)
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            # clip.write_videofile(f"calvin_seq{seq_ind}.mp4")
            # 指定输出路径
            output_path = eval_log_dir / f"calvin_seq{seq_ind}.mp4"
            clip.write_videofile(str(output_path))

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video):
    """
    Evaluates a sequence of language instructions.

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        success, video = rollout(env, model, task_checker,
                                 subtask, lang_annotation)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask, lang_annotation):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name
        lang_annotation: a string indicates the instruction of the task

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    with torch.no_grad(), torch.inference_mode():
        video = [] # show video for debugging
        obs = env.get_obs()

        model.reset()
        start_info = env.get_info()

        print('------------------------------')
        print(f'task: {lang_annotation}')
        video.append(obs["rgb_obs"]["rgb_static"])

        pbar = tqdm(range(EP_LEN))
        # from torch.amp import autocast
        for step in pbar:
            # with autocast(device_type='cuda', dtype=torch.float16):  # 新版 API
            obs = prepare_visual_states(obs, env, model.policy_cfg.n_obs_steps)
            obs = prepare_proprio_states(obs, env, model.policy_cfg.n_obs_steps)
            lang_embeddings = model.encode_instruction(lang_annotation, model.args.device)
            trajectory = model.step(obs, lang_embeddings)
            for act_ind in range(min(trajectory.shape[1], model.policy_cfg.n_action_steps)):
                # calvin_env executes absolute action in the format of:
                # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
                curr_action = [
                    trajectory[0, act_ind, :3],
                    trajectory[0, act_ind, 3:6],
                    trajectory[0, act_ind, [6]]
                ]
                pbar.set_description(f"step: {step}")
                curr_proprio = obs['proprio']
                obs, _, _, current_info = env.step(curr_action)
                obs['proprio'] = curr_proprio

                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )

                video.append(obs["rgb_obs"]["rgb_static"])

                if len(current_task_info) > 0:
                    return True, video

    return False, video


def get_calvin_gripper_loc_bounds(args):
    with open(args.calvin_gripper_loc_bounds, "r") as stream:
       bounds = yaml.safe_load(stream)
       min_bound = bounds['act_min_bound'][:3]
       max_bound = bounds['act_max_bound'][:3]
       gripper_loc_bounds = np.stack([min_bound, max_bound])

    return gripper_loc_bounds


def main(args):

    # These location bounds are extracted from every episode in play trajectory
    if args.calvin_gripper_loc_bounds is not None:
        args.calvin_gripper_loc_bounds = get_calvin_gripper_loc_bounds(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # evaluate a custom model
    workspace = get_workspace(args)
    if workspace.cfg.training.use_ema:
        model = create_model(args, workspace.ema_model, workspace.cfg)
    else:
        model = create_model(args, workspace.model, workspace.cfg)

    sequence_indices = [
        i for i in range(0, NUM_SEQUENCES)
    ]

    # 初始化环境（单卡）
    env = make_env(args.calvin_dataset_path, show_gui=False)
    
    # 评估策略
    evaluate_policy(
        model, 
        env,
        conf_dir=Path(args.calvin_model_path) / "conf",
        eval_log_dir=args.base_log_dir,
        sequence_indices=sequence_indices,
        save_video=args.save_video
    )

    # 打印结果
    results, _ = collect_results(args.base_log_dir)
    str_results = " ".join([
        f"{i + 1}/5 : {v * 100:.1f}% |" 
        for i, v in enumerate(count_success(results))
    ]) + "|"
    print(f'Evaluated {len(results)}/{NUM_SEQUENCES} episodes.')
    print(str_results)

    # 清理资源
    del env
    gc.collect()


def get_workspace(args):
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config', f'{args.config_name}.yaml'))
    cfg = OmegaConf.load(config_path)
    task_config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config', 'task', 'calvin_multi_task.yaml'))
    task_cfg = OmegaConf.load(task_config_path)
    cfg.task = task_cfg

    workspace = TrainDP3Workspace(cfg, args.output_dir)
    lastest_ckpt_path = workspace.get_checkpoint_path()
    if lastest_ckpt_path.is_file():
        print(f"Resuming from checkpoint {lastest_ckpt_path}")
        workspace.load_checkpoint(path=lastest_ckpt_path)
    return workspace


if __name__ == "__main__":
    if sys.version_info >= (3, 3):
        import collections.abc
        for attr_name in ['Mapping', 'MutableMapping', 'Sequence', 'Iterable', 'Set', 'MutableSet']:
            if not hasattr(collections, attr_name):
                setattr(collections, attr_name, getattr(collections.abc, attr_name))
    import fractions
    import math

    # Python 3.9+ fractions.gcd 修补
    if not hasattr(fractions, 'gcd'):
        fractions.gcd = math.gcd

    # 修复 np.int
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'int_'):
        np.int_ = np.int64

    # 修复 np.float
    # NumPy 兼容性修复
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'int_'):
        np.int_ = np.int64

    # 其他NumPy废弃类型修复
    np_types = {
        'bool': bool,
        'bool_': bool,
        'complex': complex,
        'complex_': np.complex128,
        'str': str,
        'str_': np.str_,
        'unicode': str,
        'unicode_': np.str_,
        'object': object,
        'object_': np.object_,
    }

    for name, dtype in np_types.items():
        if not hasattr(np, name):
            setattr(np, name, dtype)


    # args = Arguments().parse_args()
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    args = parse_args()

    # # DDP initialization
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    main(args)
