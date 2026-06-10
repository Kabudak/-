from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool


class SemanticTokenBuilder(nn.Module):
    def __init__(
        self,
        sparse_fields: list[str],
        dense_fields: list[str],
        field_embed_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.sparse_fields = sparse_fields
        self.dense_fields = dense_fields

        if sparse_fields:
            self.sparse_field_offsets = nn.Parameter(torch.zeros(len(sparse_fields), field_embed_dim))
            self.sparse_proj = nn.Sequential(
                nn.Linear(len(sparse_fields) * field_embed_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.sparse_field_offsets = None
            self.sparse_proj = None

        self.dense_proj = (
            nn.Sequential(
                nn.Linear(len(dense_fields), d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            if dense_fields
            else None
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        non_seq_sparse: torch.Tensor,
        non_seq_dense: torch.Tensor,
        sparse_embeddings: nn.ModuleDict,
        sparse_index: dict[str, int],
        dense_index: dict[str, int],
    ) -> torch.Tensor:
        batch_size = non_seq_sparse.size(0)
        device = non_seq_sparse.device
        output = torch.zeros(batch_size, self.norm.normalized_shape[0], device=device, dtype=self.norm.weight.dtype)

        if self.sparse_fields:
            sparse_parts = [
                sparse_embeddings[field](non_seq_sparse[:, sparse_index[field]])
                for field in self.sparse_fields
            ]
            sparse_stack = torch.stack(sparse_parts, dim=1)
            sparse_stack = sparse_stack + self.sparse_field_offsets.unsqueeze(0)
            sparse_repr = sparse_stack.flatten(start_dim=1)
            output = output + self.sparse_proj(sparse_repr)

        if self.dense_fields:
            dense_tensor = torch.stack(
                [non_seq_dense[:, dense_index[field]] for field in self.dense_fields],
                dim=1,
            )
            output = output + self.dense_proj(dense_tensor)

        return self.norm(output)


class StructuredSequenceStepEncoder(nn.Module):
    def __init__(
        self,
        sparse_fields: list[str],
        dense_fields: list[str],
        field_embed_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.sparse_fields = sparse_fields
        self.dense_fields = dense_fields
        if sparse_fields:
            self.sparse_field_offsets = nn.Parameter(torch.zeros(len(sparse_fields), field_embed_dim))
            self.sparse_proj = nn.Sequential(
                nn.Linear(len(sparse_fields) * field_embed_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.sparse_field_offsets = None
            self.sparse_proj = None

        self.dense_proj = (
            nn.Sequential(
                nn.Linear(len(dense_fields), d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            if dense_fields
            else None
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        seq_sparse: torch.Tensor,
        seq_dense: torch.Tensor,
        sparse_embeddings: nn.ModuleDict,
        sparse_index: dict[str, int],
        dense_index: dict[str, int],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = seq_sparse.shape
        device = seq_sparse.device
        output = torch.zeros(
            batch_size,
            seq_len,
            self.norm.normalized_shape[0],
            device=device,
            dtype=self.norm.weight.dtype,
        )

        if self.sparse_fields:
            sparse_parts = [
                sparse_embeddings[field](seq_sparse[:, :, sparse_index[field]])
                for field in self.sparse_fields
            ]
            sparse_stack = torch.stack(sparse_parts, dim=2)
            sparse_stack = sparse_stack + self.sparse_field_offsets.view(1, 1, len(self.sparse_fields), -1)
            sparse_repr = sparse_stack.flatten(start_dim=2)
            output = output + self.sparse_proj(sparse_repr)

        if self.dense_fields:
            dense_tensor = torch.stack(
                [seq_dense[:, :, dense_index[field]] for field in self.dense_fields],
                dim=2,
            )
            output = output + self.dense_proj(dense_tensor)

        return self.norm(output)


class TAACHyFormerClassifier(nn.Module):
    def __init__(
        self,
        sparse_field_cardinalities: dict[str, int],
        non_seq_sparse_fields: list[str],
        non_seq_dense_fields: list[str],
        seq_sparse_fields: list[str],
        seq_dense_fields: list[str],
        token_groups: dict[str, list[str]],
        num_classes: int,
        seq_len: int,
        num_sequences: int,
        num_non_seq_tokens: int,
        num_queries_per_seq: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        hyformer_layers: int,
        seq_encoder_type: str = "longer",
        short_seq_len: int = 8,
        field_embed_dim: int = 24,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.num_queries_per_seq = num_queries_per_seq
        self.total_query_tokens = num_sequences * num_queries_per_seq
        self.d_model = d_model
        self.token_groups = token_groups

        self.non_seq_sparse_index = {name: idx for idx, name in enumerate(non_seq_sparse_fields)}
        self.non_seq_dense_index = {name: idx for idx, name in enumerate(non_seq_dense_fields)}
        self.seq_sparse_index = {name: idx for idx, name in enumerate(seq_sparse_fields)}
        self.seq_dense_index = {name: idx for idx, name in enumerate(seq_dense_fields)}

        self.sparse_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(cardinality, field_embed_dim, padding_idx=0)
                for field, cardinality in sparse_field_cardinalities.items()
            }
        )

        self.semantic_token_builders = nn.ModuleDict()
        for token_name, source_fields in token_groups.items():
            sparse_fields = [field for field in source_fields if field in self.non_seq_sparse_index]
            dense_fields = [field for field in source_fields if field in self.non_seq_dense_index]
            self.semantic_token_builders[token_name] = SemanticTokenBuilder(
                sparse_fields=sparse_fields,
                dense_fields=dense_fields,
                field_embed_dim=field_embed_dim,
                d_model=d_model,
            )

        self.num_non_seq_tokens = len(self.semantic_token_builders)
        if num_non_seq_tokens != self.num_non_seq_tokens:
            raise ValueError(
                f"num_non_seq_tokens={num_non_seq_tokens}"
                f"{self.num_non_seq_tokens}"
            )

        self.sequence_step_encoder = StructuredSequenceStepEncoder(
            sparse_fields=seq_sparse_fields,
            dense_fields=seq_dense_fields,
            field_embed_dim=field_embed_dim,
            d_model=d_model,
        )
        self.sequence_position_embedding = nn.Embedding(seq_len, d_model)
        self.sequence_type_embedding = nn.Embedding(num_sequences, d_model)

        # Query generation uses the non-seq token set plus all pooled sequence
        # summaries, so every branch starts from a multi-sequence global context.
        query_input_dim = (self.num_non_seq_tokens + num_sequences) * d_model
        self.query_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(query_input_dim, ffn_hidden),
                    nn.SiLU(),
                    nn.Linear(ffn_hidden, num_queries_per_seq * d_model),
                )
                for _ in range(num_sequences)
            ]
        )

        self.backbone = HyFormerBackbone(
            num_layers=hyformer_layers,
            num_sequences=num_sequences,
            num_queries_per_sequence=num_queries_per_seq,
            num_non_seq_tokens=self.num_non_seq_tokens,
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            encoder_type=seq_encoder_type,
            short_seq_len=short_seq_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes),
        )

    def build_non_seq_tokens(
        self,
        non_seq_sparse: torch.Tensor,
        non_seq_dense: torch.Tensor,
    ) -> torch.Tensor:
        tokens = [
            builder(
                non_seq_sparse=non_seq_sparse,
                non_seq_dense=non_seq_dense,
                sparse_embeddings=self.sparse_embeddings,
                sparse_index=self.non_seq_sparse_index,
                dense_index=self.non_seq_dense_index,
            )
            for builder in self.semantic_token_builders.values()
        ]
        return torch.stack(tokens, dim=1)

    def build_sequence_tokens(
        self,
        seq_sparse: torch.Tensor,
        seq_dense: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequence_tokens: list[torch.Tensor] = []
        sequence_masks: list[torch.Tensor] = []
        pooled_sequences: list[torch.Tensor] = []

        for seq_idx in range(self.num_sequences):
            current_tokens = self.sequence_step_encoder(
                seq_sparse=seq_sparse[:, seq_idx, :, :],
                seq_dense=seq_dense[:, seq_idx, :, :],
                sparse_embeddings=self.sparse_embeddings,
                sparse_index=self.seq_sparse_index,
                dense_index=self.seq_dense_index,
            )
            position_ids = torch.arange(current_tokens.size(1), device=current_tokens.device)
            current_tokens = current_tokens + self.sequence_position_embedding(position_ids).unsqueeze(0)
            current_tokens = current_tokens + self.sequence_type_embedding.weight[seq_idx].view(1, 1, -1)
            current_mask = ensure_non_empty_mask(seq_mask[:, seq_idx, :])
            sequence_tokens.append(current_tokens)
            sequence_masks.append(current_mask)
            pooled_sequences.append(masked_mean_pool(current_tokens, current_mask))

        return sequence_tokens, sequence_masks, pooled_sequences

    def build_query_tokens(
        self,
        non_seq_tokens: torch.Tensor,
        pooled_sequences: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(pooled_sequences) != self.num_sequences:
            raise ValueError(
                f"Expected {self.num_sequences} pooled sequence summaries, but received {len(pooled_sequences)}"
            )

        batch_size = non_seq_tokens.size(0)
        ns_flat = non_seq_tokens.flatten(start_dim=1)
        sequence_context = torch.cat(pooled_sequences, dim=-1)
        global_context = torch.cat([ns_flat, sequence_context], dim=-1)
        query_tokens: list[torch.Tensor] = []
        for seq_idx in range(self.num_sequences):
            query_tokens.append(
                self.query_generators[seq_idx](global_context).view(
                    batch_size,
                    self.num_queries_per_seq,
                    self.d_model,
                )
            )
        return query_tokens

    def forward(
        self,
        non_seq_sparse: torch.Tensor,
        non_seq_dense: torch.Tensor,
        seq_sparse: torch.Tensor,
        seq_dense: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        non_seq_tokens = self.build_non_seq_tokens(non_seq_sparse, non_seq_dense)
        sequence_tokens, sequence_masks, pooled_sequences = self.build_sequence_tokens(seq_sparse, seq_dense, seq_mask)
        query_tokens = self.build_query_tokens(non_seq_tokens, pooled_sequences)
        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        query_repr = boosted_tokens[:, : self.total_query_tokens, :].mean(dim=1)
        non_seq_repr = boosted_tokens[:, self.total_query_tokens :, :].mean(dim=1)
        return self.head(torch.cat([query_repr, non_seq_repr], dim=-1))
