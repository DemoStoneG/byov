#ckpt_dir='ckpt_path'

# use extracted embeddings (saved in ckpt_dir/$dataset_eval)
for dataset in break_eggs pour_milk pour_liquid tennis_forehand; do
    python evaluation/evaluate_features.py --dataset $dataset \
        --eval_task 1234 \
        --ckpt /mnt/data/wzh/projects/byov-main/logs/exp_break_eggs/bestconfig/lightning_logs/version_2/checkpoints/epoch=299.ckpt \
        --extract_embedding
done
