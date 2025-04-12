#!/bin/zsh
# 加载 conda 初始化脚本（针对 zsh 环境）
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

conda activate conflict
# [Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-2-7b-chat, Llama-2-13b-chat, 
# Llama-2-70b-chat, falcon-7b, falcon-40b, falcon-7b-instruct, 
# falcon-40b-instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1]
test_model=MOF

# [trivia_qa, squad, bioasq, nq, svamp]
dataset=mmvp
HF_ENDPOINT=https://hf-mirror.com
image_path_list=(/data/public/datasets/image_no_correct_labels /data/public/datasets/image_with_correct_labels)
for image_path in "${image_path_list[@]}"
do
    /data/huangtao/miniconda3/envs/conflict/bin/python ../generate_answers.py \
        --model_name "$test_model" \
        --dataset "$dataset" \
        --image_path "$image_path"
done




