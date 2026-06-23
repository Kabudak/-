# HyFormer Commands

## Production Parquet Sample

Preprocess the 100-row production sample into tensors:

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0_final_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 100 --non-seq-bag-len 64
```

Train a small HyFormer run on the processed tensors:

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 32 --ffn-hidden 64 --hyformer-layers 1 --short-seq-len 8
```

The production pipeline now reads the binary label from `label_click`.
The manual feature file builds two non-sequence tokens (`context_token`, `item_token`) and three sequence branches (`click_seq`, `impression_seq`, `buy_seq`).
Non-sequence sparse array features are saved as sparse bags and mean-pooled inside the model.

## Public Taobao Ad Data

Preprocess the public CSV data:

```bash
python scripts/preprocess_baotao.py --data-dir data/archive --output-dir data/processed --seq-len 100 --num-sequences 2
```

Train on the public preprocessed tensors:

```bash
python scripts/run_baotao.py --data-dir data/processed --seq-encoder-type longer --save-checkpoint
```
