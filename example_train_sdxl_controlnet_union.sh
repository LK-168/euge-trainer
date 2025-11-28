#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=3,4}"                  # 指定 GPU
: "${OUTPUT_DIR:=/data2/xujr/output/union_test01}"  # 输出目录
: "${UNION_JSON_PATH:=/data2/xujr/anicontrol-20k/union_manifest.json}" # 由工具生成的 JSON
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=1}"
: "${LR:=1e-5}"
: "${EPOCHS:=1}"
: "${SAVE_EVERY_STEPS:=1000}"
: "${EVAL_EVERY_STEPS:=1000}"


######################
# 运行信息打印        #
######################
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Union JSON: ${UNION_JSON_PATH}"
echo "Batch size: ${BATCH_SIZE}, Grad Accum: ${GRAD_ACCUM}, LR: ${LR}, Epochs: ${EPOCHS}"


######################
# 启动训练（多 GPU）  #
######################

NUM_PROCESSES=${NUM_PROCESSES:-2}

accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --multi_gpu \
  train.py \
  --trainer sdxl_controlnet_union \
  --config ./configs/example_config_train_sdxl_controlnet_union.py \
  --config.pretrained_model_name_or_path="/data2/xujr/model/noobaiXLNAIXL_epsilonPred11Version/noobaiXLNAIXL_epsilonPred11Version.safetensors" \
  --config.output_dir="${OUTPUT_DIR}" \
  --config.union_json_path="${UNION_JSON_PATH}" \
  --config.batch_size=${BATCH_SIZE} \
  --config.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.learning_rate_controlnet=${LR} \
  --config.num_train_epochs=${EPOCHS} \
  --config.save_every_n_steps=${SAVE_EVERY_STEPS} \
  --config.eval_every_n_steps=${EVAL_EVERY_STEPS} \
  --config.train_controlnet=True \
  --config.train_nnet=False \
  --config.mixed_precision="fp16" \
  --config.full_fp16=True \
  --config.xformers=True \
  --config.gradient_checkpointing=True

echo "\n✅ SDXL ControlNet Union 多 GPU 训练已启动（进程数：${NUM_PROCESSES}）。日志与模型将保存在：${OUTPUT_DIR}"
