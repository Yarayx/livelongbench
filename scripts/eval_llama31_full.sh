export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=meta-llama/Meta-Llama-3.1-8B-Instruct
max_length=50000
strategy=""
result_dir=llama31/full
n_bit=4

python -m tasks.eval_LongLive \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 32 \
            --max_length $max_length \

