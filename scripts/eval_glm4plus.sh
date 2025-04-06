#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

max_length=50000
result_dir=glm4plus
models=gpt4o.yaml

python -m tasks.eval_LongLive_api \
            --pipeline LongLive \
            --result_dir $result_dir \
            --models $models \
