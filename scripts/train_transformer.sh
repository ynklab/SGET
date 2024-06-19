seed=$1

mkdir -p ../models/Transformer/$seed/checkpoints
mkdir -p ../models/Transformer/$seed/logs

onmt_build_vocab -config config_transformer.yaml -n_sample -1
onmt_train -config config_transformer.yaml