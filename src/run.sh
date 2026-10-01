# python -m augmentation.prepare_dataset_description \
#   --max-tokens 1024 \
#  --input data/pyra_desc_yosys_filtered \
#  --output data/pyra_desc_qwen \
#  --model-name Qwen/Qwen3-Coder-Next \
#  --api-backend local-vllm \
#  --concurrency 64 \
#  --local-vllm-batch-size 64

python -m augmentation.run_data_augmentation \
  --input data/pyra_desc_qwen \
  --output data/pyra_augmentation_random_000_199 \
  --work-dir outputs/augmentation/eoh_pyra_random_000_199 \
  --targets area power delay \
  --target-selection-mode random \
  --num-generations 6 \
  --candidates-per-generation 5 \
  --start 0 \
  --limit 300 \
  --api-backend local-vllm \
  --model-name Qwen/Qwen3-Coder-Next \
  --concurrency 64 \
  --llm-concurrency 64 \
  --synthesis-concurrency 8 \
  --local-vllm-batch-size 64 \
  --local-vllm-batch-timeout-ms 50 \
  --local-vllm-gpu-memory-utilization 0.92 \
  --local-vllm-max-num-seqs 64 \
  --max-runtime-hours 47.5 \
  --shutdown-margin-minutes 15
