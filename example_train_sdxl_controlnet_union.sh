#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG=INFO
# export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/LK/hf_cache

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"              # 指定 GPU
OUTPUT_DIR=/root/LK/output/union_test09_2tp
UNION_JSON_PATH=/root/LK/anicontrol-20k/union_manifest_2tp.json 
CONFIG_PATH="./configs/example_config_train_sdxl_controlnet_union.py"
BASE_MODEL_PATH="/root/LK/noobai-XL-1.1"

BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-15}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-200}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-500}
NUM_PROCESSES=${NUM_PROCESSES:-2}   # 使用的 GPU 数量

extra_parameters=()



# 如果使用多个 GPU，则添加“--multi_gpu”参数
if [ "${NUM_PROCESSES}" -gt 1 ]; then
    extra_parameters+=("--multi_gpu")
fi

######################
# 运行信息打印        #
######################
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Union JSON: ${UNION_JSON_PATH}"
echo "Batch size: ${BATCH_SIZE}, LR: ${LR}, Epochs: ${EPOCHS}"


######################
# 启动训练（多 GPU）  #
######################



accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  "${extra_parameters[@]}" \
  train.py \
  --trainer sdxl_controlnet_union \
  --config ${CONFIG_PATH} \
  --config.pretrained_model_name_or_path="${BASE_MODEL_PATH}" \
  --config.output_dir="${OUTPUT_DIR}" \
  --config.union_json_path="${UNION_JSON_PATH}" \
  --config.batch_size=${BATCH_SIZE} \
  --config.learning_rate_controlnet=${LR} \
  --config.num_train_epochs=${EPOCHS} \
  --config.save_every_n_steps=${SAVE_EVERY_STEPS} \
  --config.eval_every_n_steps=${EVAL_EVERY_STEPS} \
  --config.train_controlnet=True \
  --config.train_nnet=False \
  --config.xformers=True \
  --config.gradient_checkpointing=True

echo "\n✅ SDXL ControlNet Union 多 GPU 训练已启动（进程数：${NUM_PROCESSES}）。日志与模型将保存在：${OUTPUT_DIR}"
