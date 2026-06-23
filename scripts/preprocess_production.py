from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.production_data import build_tensors, load_parquet_columns


def parse_sequence_lens(raw_value: str | None) -> dict[str, int] | None:
    if raw_value is None or not raw_value.strip():
        return None
    result: dict[str, int] = {}
    for part in raw_value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid sequence length item: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = int(value.strip())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize production parquet features for HyFormer")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "000000_0.gz_head100.parquet",
        help="Production parquet file to read.",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "selectedfeaturefinal.txt",
        help="Feature grouping file with manually curated groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "production_sample",
        help="Directory where tensor files and metadata.json will be written.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick checks.")
    parser.add_argument("--seq-len", type=int, default=100, help="Maximum length per behavior sequence.")
    parser.add_argument(
        "--sequence-lens",
        default=None,
        help="Optional comma-separated per-branch lengths, for example click_seq=100,impression_seq=10,buy_seq=10.",
    )
    parser.add_argument("--non-seq-bag-len", type=int, default=64, help="Maximum length per non-sequence sparse bag feature.")
    parser.add_argument(
        "--non-seq-array-reduction",
        choices=("last", "mean"),
        default="last",
        help="How dense non-sequence array features are reduced into one scalar feature.",
    )
    parser.add_argument(
        "--sequence-truncation",
        choices=("head", "tail"),
        default="tail",
        help="Keep the first or last seq-len items when a behavior list is longer than seq-len.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] reading parquet: {args.input_parquet}")
    columns, arrow_types = load_parquet_columns(args.input_parquet, max_rows=args.max_rows)

    print(f"[data] feature schema: {args.feature_file}")
    sequence_lens = parse_sequence_lens(args.sequence_lens)
    tensors = build_tensors(
        columns=columns,
        feature_file=args.feature_file,
        seq_len=args.seq_len,
        sequence_truncation=args.sequence_truncation,
        non_seq_bag_len=args.non_seq_bag_len,
        sequence_lens=sequence_lens,
        non_seq_array_reduction=args.non_seq_array_reduction,
    )
    (
        non_seq_sparse,
        non_seq_sparse_bag,
        non_seq_sparse_bag_mask,
        non_seq_dense,
        seq_sparse,
        seq_dense,
        seq_mask,
        labels,
        metadata,
    ) = tensors

    torch.save(non_seq_sparse, args.output_dir / "non_seq_sparse.pt")
    torch.save(non_seq_sparse_bag, args.output_dir / "non_seq_sparse_bag.pt")
    torch.save(non_seq_sparse_bag_mask, args.output_dir / "non_seq_sparse_bag_mask.pt")
    torch.save(non_seq_dense, args.output_dir / "non_seq_dense.pt")
    torch.save(seq_sparse, args.output_dir / "seq_sparse.pt")
    torch.save(seq_dense, args.output_dir / "seq_dense.pt")
    torch.save(seq_mask, args.output_dir / "seq_mask.pt")
    torch.save(labels, args.output_dir / "labels.pt")

    metadata["source_parquet"] = str(args.input_parquet)
    metadata["feature_file"] = str(args.feature_file)
    metadata["preprocess_args"] = {
        "max_rows": args.max_rows,
        "seq_len": args.seq_len,
        "sequence_lens": sequence_lens,
        "non_seq_bag_len": args.non_seq_bag_len,
        "non_seq_array_reduction": args.non_seq_array_reduction,
        "sequence_truncation": args.sequence_truncation,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[data] saved tensors")
    print(f"  output_dir:       {args.output_dir}")
    print(f"  labels:           {tuple(labels.shape)}  pos_rate={metadata['pos_rate']:.4f}")
    print(f"  non_seq_sparse:   {tuple(non_seq_sparse.shape)}")
    print(f"  non_seq_bag:      {tuple(non_seq_sparse_bag.shape)}")
    print(f"  non_seq_bag_mask: {tuple(non_seq_sparse_bag_mask.shape)}")
    print(f"  non_seq_dense:    {tuple(non_seq_dense.shape)}")
    print(f"  seq_sparse:       {tuple(seq_sparse.shape)}")
    print(f"  seq_dense:        {tuple(seq_dense.shape)}")
    print(f"  seq_mask:         {tuple(seq_mask.shape)}")
    print(f"  sequence_names:   {metadata['sequence_names']}")
    print(f"  sequence_lens:    {metadata['sequence_lens']}")
    print(f"  sequence_backbone:{metadata['sequence_backbone_fields']}")
    print(f"  token_groups:     {list(metadata['token_groups'].keys())}")


if __name__ == "__main__":
    main()
