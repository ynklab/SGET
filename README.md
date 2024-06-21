# SGET
This repository contains SGET, **S**tructural **GE**neralization Benchmark based on English-Japanese Machine **T**ranslation.

## Dataset
SGET is in `data/`.
`data/train.txt` is used for training models, and `data/dev.txt` is used for validations.
`data/test.txt` is the in-distribution test set and `data/gen.txt` is the out-of-distribution generalization set.

## Experiments

### Setup
```
pip install -r requirements.txt
```

### Train
For training vanilla Transformer, first replace `[SEED]` in `scripts/config_transformer.yaml` with a random seed.
Then run the following command, replacing `[SEED]` with the seed.
```
cd scripts
sh train_transformer.sh [SEED]
```
You can train LSTM (`lstm`) similarly.

For fine-tuning Llama 2, run the following command.
```
cd scripts
sh lora_llama2.sh [SEED]
```

### Translate
Run the following command for generating translations using a trained (fine-tuned) model.
The checkpoint used for translations is determined by `[SEED]` and `[TRAIN_STEPS]`.
```
cd scripts
sh translate_transformer.sh [SEED] [TRAIN_STEPS]
```
You can use LSTM and Llama 2 similarly.

Translation results will be generated in `results/[MODEL]/[SEED]/[TRAIN_STEPS]`.

## License
This repository is primarily licensed under MIT License, but `scripts/lora.py` is licensed under Apache License, Version 2.0.

## Citation
You can reference this work as follows:
```
@article{kumon-etal-2024-structural,
    title = {Evaluating Structural Generalization in Neural Machine Translation},
    author = {Kumon, Ryoma and Matsuoka, Daiki and Yanaka, Hitomi},
    journal={arXiv preprint arXiv:2406.13363},
    year = {2024},
}
```

## Contact
If you have any issues or questions, please contact [kumoryo9@is.s.u-tokyo.ac.jp](mailto:kumoryo9@is.s.u-tokyo.ac.jp)