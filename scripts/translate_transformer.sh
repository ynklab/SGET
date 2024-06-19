SEED=$1
STEP_NUM=$2

mkdir -p ../results/Transformer/$SEED/$STEP_NUM

onmt_translate -model ../models/Transformer/$SEED/checkpoints/model_step_$STEP_NUM.pt\
        -src ../data/bpe/test_en_encoded_sp.txt \
        -tgt ../data/bpe/test_ja_encoded_sp.txt \
        -replace_unk \
        -output ../results/Transformer/$SEED/$STEP_NUM/pred_test.txt -gpu 0 -batch_size 128  -verbose
onmt_translate -model ../models/Transformer/$SEED/checkpoints/model_step_$STEP_NUM.pt \
        -src ../data/bpe/gen_en_encoded_sp.txt \
        -tgt ../data/bpe/gen_ja_encoded_sp.txt \
        -replace_unk -output ../results/Transformer/$SEED/$STEP_NUM/pred_gen.txt -gpu 0 -batch_size 128 -verbose
