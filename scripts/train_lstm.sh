SEED=$1

mkdir -p ../models/LSTM/$SEED/checkpoints
mkdir -p ../models/LSTM/$SEED/logs

onmt_build_vocab -config config_lstm.yaml -n_sample -1
onmt_train -config config_lstm.yaml