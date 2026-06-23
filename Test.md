# HyFormer Commands

## Production Parquet Sample

Preprocess the 100-row production sample into tensors:

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0.gz_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 200 --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 --non-seq-bag-len 64 --non-seq-array-reduction last
```

Train a small HyFormer run on the processed tensors:

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 128 --field-embed-dim 64 --token-mlp-hidden 320 --ffn-hidden 256 --hyformer-layers 1 --short-seq-len 8
```

The production pipeline now reads the binary label from `label_click`.
The manual feature file builds two non-sequence tokens (`context_token`, `item_token`) and three sequence branches (`click_seq`, `impression_seq`, `buy_seq`).
Each sequence branch uses its first selected field as the branch backbone for mask construction.
Non-sequence dense arrays are reduced into one scalar feature instead of being expanded into multiple summary features.
Non-sequence sparse array features are saved as sparse bags and mean-pooled inside the model.
Feature vectors are embedded to `field_embed_dim` first, concatenated inside each group, and projected to `d_model` by a shared MLP.

## Public Taobao Ad Data

Preprocess the public CSV data:

```bash
python scripts/preprocess_baotao.py --data-dir data/archive --output-dir data/processed --seq-len 100 --num-sequences 2
```

Train on the public preprocessed tensors:

```bash
python scripts/run_baotao.py --data-dir data/processed --seq-encoder-type longer --save-checkpoint
```
