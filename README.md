# Reinforcement Learning with Temporal and Variable Dependency-aware Transformer for Stock Trading Optimization

## Preparation

### Installation
```
pip install -r requirements.txt
```

### Dataset
Downloaded from [YahooFinance](https://pypi.org/project/yfinance/)

## Experiment

### Data 
dir: '*data/*' 
```
data/
├── CSI/     # CSI-300  
├── SP/      # S&P-100
└── NASDAQ/  # NASDAQ-100
```
### Code

dir:'*code/*'

```
code/
├── Transformer/
│   └── TVDT/                   # the Temporal and Variable Dependency-aware Transformer
│       ├── two_stage_attn.py          # the two-stage attention mechanism
│       ├── variable_embed.py          # the variable embedding
│       └── ...                        # other components of TVDT
│    └── script/
│        ├── train_pred_long.sh
│        ├── train_pred_short.sh
│        └── train_mae.sh
└── MySAC/
    └── SAC/
        └── policy_transformer.py      # the dual adaptive attention mechanism
└── train_rl.py
```
The above directory structure highlights the Python files containing our key innovations.
### Training
#### 1st stage：Representation Learning

1）Relation representation module training: 

```bash
cd code/Transformer/script
sh train_mae.sh
```

2）Long-term prediction module training:

```bash
cd code/Transformer/script
sh train_pred_long.sh
```

3) Short-term prediction module training:

```bash
cd code/Transformer/script
sh train_pred_short.sh
```

4) Select the best model of three representation modules from '*code/Transformer/checkpoints/*' according to their performance on validation set and add them to '*code/Transformer/pretrained/*'

**OR** directly use the model which have been pretrained in advance by us (dir:'*code/Transformer/pretrained/sp_100/* ')

#### 2nd stage：Policy learning

1) Policy decision module training (three representation learning module's path can be changed in *train_rl.py* file)

```bash
python train_rl.py
```

2) get policy optimization result on test set from '*code/results/df_print/*'

[//]: # (## Citation)


## Acknowledgements
This codebase is based on [StockFormer](https://github.com/gsyyysg/StockFormer).

We sincerely thank the authors of Stockformer for their valuable contribution;
our code was developed based on their work.
