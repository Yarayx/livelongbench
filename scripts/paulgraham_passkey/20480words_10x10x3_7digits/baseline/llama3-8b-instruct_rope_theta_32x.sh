output_dir_root="./data"
# scripts/paulgraham_passkey/20480words_10x10x3_7digits/baseline/llama3-8b-instruct_rope_theta_32x.sh
task="paulgraham_passkey"
dataset="20480words_10x10x3_7digits"
model="llama3-8b-instruct_rope_theta_32x"
method="baseline"
n_bit=4
python pipeline_passkey/${method}/main.py \
--exp_desc ${task}_${dataset}_${model}_${method} \
--pipeline_config_dir config/pipeline_config/${method}/${task}/${model}.json \
--eval_config_dir config/eval_config/${task}/${dataset}.json \
--output_folder_dir ${output_dir_root}/${task}/${dataset}/${method}_minf/${model}/ \
# --cache_implementation quantized \
# --cache_backend quanto \
# --cache_nbits $n_bit \
# --load_in_4bit 
