# HyFormer Commands

## Production Parquet Sample

Preprocess the 100-row production sample into tensors:

```bash
python scripts/preprocess_production.py --input-parquet 000000_0_selected_head100.parquet --output-dir data/production_sample --seq-len 100
```

Train a small HyFormer run on the processed tensors:

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 32 --ffn-hidden 64 --hyformer-layers 1 --short-seq-len 8
```

Useful label variants:

```bash
python scripts/preprocess_production.py --label-mode rel_level_positive
python scripts/preprocess_production.py --label-mode rel_score_present
```

`rel_score_present` is the default. It treats `rel_score_bkt >= 0` as positive and `-1` or missing as negative.

## Public Taobao Ad Data

Preprocess the public CSV data:

```bash
python scripts/preprocess_baotao.py --data-dir data/archive --output-dir data/processed --seq-len 100 --num-sequences 2
```

Train on the public preprocessed tensors:

```bash
python scripts/run_baotao.py --data-dir data/processed --seq-encoder-type longer --save-checkpoint
```
