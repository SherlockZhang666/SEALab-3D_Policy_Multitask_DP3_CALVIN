# Examples:
# bash scripts/eval_policy_calvin.sh instr_dp3 calvin_multi_task 0721 0 0


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    # wandb_mode=online
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

export PYTHONPATH=/data/sea_disk0/zhangxx/3DDA_DP3:$PYTHONPATH

# torchrun --nproc_per_node 1 --master_port $RANDOM eval_calvin.py \
#     --config_name ${config_name} \
#     --calvin_gripper_loc_bounds /data/sea_disk0/zhangxx/3DDA_DP3/calvin/dataset/calvin_debug_dataset/validation/statistics.yaml \
#     --save_video 1 \
#     --output_dir ${run_dir}

python eval_calvin_one_card.py \
    --config_name ${config_name} \
    --calvin_gripper_loc_bounds /data/sea_disk0/zhangxx/calvin/dataset/task_ABC_D/validation/statistics.yaml \
    --save_video 1 \
    --output_dir ${run_dir}
                                