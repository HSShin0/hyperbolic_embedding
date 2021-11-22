# Hyperbolic Embedding
An attempt to reproduce the results in [Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations).
Still WIP.

## Getting started

### Environment
* python 3.9
* nltk==3.5
* numpy==1.21.2
* pandas==1.3.4
* scikit-learn==1.0.1
* scipy==1.7.1
* torch==1.10.0
* tqdm==4.62.3
* wandb==0.12.1

### Datasets
Use the [script in the original repo](https://github.com/facebookresearch/poincare-embeddings/blob/ff1d846db3a64a759e56173d7846c164a37654f9/wordnet/transitive_closure.py) for constructing `mammal_closure.csv`, which is the subtree of `mammal` in the tree of all nouns in WordNet.

Dataset directory structure:
```
datasets
└── wordnet
    └── mammal_closure.csv
```

## Usages

### Train
Train the embedding of `mammal` tree to **Poincare disk**.

```sh
$ python main.py
$ python main.py --help
usage: main.py [-h] [--space-type {euclid,poincare}] [--datapath DATAPATH] [--n-neg N_NEG]
               [--batch-size BATCH_SIZE] [--emb-dim EMB_DIM] [--epochs EPOCHS] [--eval-every EVAL_EVERY]
               [--exp-root EXP_ROOT] [--wlog] [--wlog-name WLOG_NAME] [--use-gpu] [--init-lr INIT_LR]
               [--burn-in-epochs BURN_IN_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --space-type {euclid,poincare}
                        embbeding space type (default: poincare)
  --datapath DATAPATH   Data csv filepath. (default: ./datasets/wordnet/mammal_closure.csv)
  --n-neg N_NEG         Number of negative samples for each positive pair. (default: 10)
  --batch-size BATCH_SIZE
                        Batch size for training. (default: 32)
  --emb-dim EMB_DIM     Embedding dimension. (default: 2)
  --epochs EPOCHS       Number of training epochs. (default: 10)
  --eval-every EVAL_EVERY
                        Evaluate every `eval_every`-epochs. (default: 5)
  --exp-root EXP_ROOT   Root dir for saving checkpoints. (default: exp/temp)
  --wlog                Use WanDB logger (default: False)
  --wlog-name WLOG_NAME
                        Run ID in WanDB logger. (default: )
  --use-gpu             Use GPU (default: False)
  --init-lr INIT_LR, -lr INIT_LR
                        Initial learning rate (default: 0.01)
  --burn-in-epochs BURN_IN_EPOCHS, -be BURN_IN_EPOCHS
                        Number of `burn-in` epochs. Use smaller learning rate. (default: 10)
```

## Issues and TODO list
- [ ] Revisit the negative sampling
- [ ] Revisit the numerical unstability issue, fixed in an ad-hoc way
- [ ] Implement the baseline model (i.e., Euclidean embedding)
- [ ] Implement Mean Rank evaluation
- [ ] Visualize the result
- [ ] Reproduce the results in the paper
- [ ] Fix speed issue for `num_workers > 0` setting of dataloader


## References
- Original paper: [Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations)
- Original code: https://github.com/facebookresearch/poincare-embeddings
