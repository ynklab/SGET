SEED=$1
STEP_NUM=$2

mkdir -p ../results/LSTM/$SEED/$STEP_NUM
onmt_translate -model ../models/LSTM/$SEED/checkpoints/model_step_$STEP_NUM.pt\
        -src ../data/bpe/test_en_encoded_sp.txt \
        -tgt ../data/bpe/test_ja_encoded_sp.txt \
        -output ../results/LSTM/$SEED/$STEP_NUM/pred_test.txt -gpu 0 -batch_size 128 -verbose
onmt_translate -model ../models/LSTM/$SEED/checkpoints/model_step_$STEP_NUM.pt \
        -src ../data/bpe/gen_en_encoded_sp.txt \
        -tgt ../data/bpe/gen_ja_encoded_sp.txt \
        -output ../results/LSTM/$SEED/$STEP_NUM/pred_gen.txt -gpu 0 -batch_size 128 -verbose
