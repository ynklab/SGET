SEED=$1
NUM_STEPS=$2

mkdir -p ../results/Llama2/$SEED/
python translate.py --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path '../data/json/test.json' \
    --lora_path ../models/Llama2/$SEED/checkpoint-$NUM_STEPS \
    --save_path ../results/Llama2/$SEED/pred_test.txt
python translate.py --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path '../data/json/gen.json' \
    --lora_path ../models/Llama2/$SEED/checkpoint-$NUM_STEPS \
    --save_path ../results/Llama2/$SEED/pred_gen.txt