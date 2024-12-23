# Examples:

# bash scripts/train_policy.sh idp3 gr1_dex-3d 1213+1211_40 1213+1211
# bash scripts/train_policy.sh dp_224x224_r3m gr1_dex-image 1213+1211_40_2d 1213+1211_rgb
# bash scripts/train_policy.sh dp_lgb21 dp_dex 1219_30_2i 1219
# bash scripts/train_policy.sh dp_224x224_r3m dp_dex 1219_30_2i 1219

# dataset_path=/storage/liujinxin/code/ArmRobot/training_data_example # dim: 32


alg_name=${1}
task_name=${2}
addition_info=${3}
dataset_path="/storage/liujinxin/code/ArmRobot/dataset/train_data/${4}"

echo "dataset: $dataset_path"

current_time=$(date +"%-m-%-d-%-H-%-M-%-S")
echo "Current time: $current_time"

DEBUG=False
wandb_mode=online #online #offline


config_name=${alg_name}

seed=0
exp_name=${alg_name}_${addition_info}_$current_time
echo "exp_name: $exp_name"
run_dir="outputs/${exp_name}_seed${seed}"

echo "run_dir: $run_dir"
echo

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd Improved-3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 

