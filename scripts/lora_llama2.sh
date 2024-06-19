SEED=$1

mkdir -p ../models/Llama2/$SEED

python lora.py --base_model 'meta-llama/Llama-2-7b-hf' \
    --train_path '../data/json/train.json' \
    --dev_path '../data/json/dev.json' \
    --output_dir "../models/Llama2/$SEED" \
    --num_epochs 8 \
    --batch_size 64 \
    --micro_batch_size 64 \
    --val_set_size 1 \
    --learning_rate 1e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed $SEED