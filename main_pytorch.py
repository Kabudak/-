from __future__ import annotations

import math

import torch
from torch import nn

VALID_ENCODER_TYPES = {"longer", "full_transformer", "swiglu"}


def ensure_non_empty_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype is not torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch_size, seq_len]")
    if mask.size(1) == 0:
        raise ValueError("mask sequence length must be positive")
    if mask.all(dim=1).all():
        return mask

    fixed_mask = mask.clone()
    empty_rows = ~fixed_mask.any(dim=1)
    fixed_mask[empty_rows, 0] = True
    return fixed_mask


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_non_empty_mask(mask)
    weights = mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def build_window_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    batch_size, seq_len = mask.shape
    pooled_mask = torch.zeros(batch_size, target_len, device=mask.device, dtype=torch.bool)
    for idx in range(target_len):
        start = math.floor(idx * seq_len / target_len)
        end = max(start + 1, math.floor((idx + 1) * seq_len / target_len))
        pooled_mask[:, idx] = mask[:, start:end].any(dim=1)
    return ensure_non_empty_mask(pooled_mask)


def masked_chunk_pool(x: torch.Tensor, mask: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ensure_non_empty_mask(mask)
    batch_size, seq_len, d_model = x.shape
    target_len = max(1, min(target_len, seq_len))
    pooled = torch.zeros(batch_size, target_len, d_model, device=x.device, dtype=x.dtype)

    for idx in range(target_len):
        start = math.floor(idx * seq_len / target_len)
        end = max(start + 1, math.floor((idx + 1) * seq_len / target_len))
        pooled[:, idx, :] = masked_mean_pool(x[:, start:end, :], mask[:, start:end])

    pooled_mask = build_window_mask(mask, target_len)
    return pooled, pooled_mask


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.proj(x).chunk(2, dim=-1)
        return self.out(torch.nn.functional.silu(gate) * value)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.kv_norm(key_value),
            self.kv_norm(key_value),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return query + attn_out


class FullTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.norm_0 = nn.LayerNorm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = SwiGLUFeedForward(d_model, ffn_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~ensure_non_empty_mask(mask)
        attn_out, _ = self.attn(
            self.norm_0(x),
            self.norm_0(x),
            self.norm_0(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_1(x))
        return x, ~key_padding_mask


class LongerStyleEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, short_seq_len: int) -> None:
        super().__init__()
        self.short_seq_len = short_seq_len
        self.cross_attn = CrossAttentionBlock(d_model=d_model, num_heads=num_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, ffn_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        short_x, short_mask = masked_chunk_pool(x, mask, self.short_seq_len)
        key_padding_mask = ~ensure_non_empty_mask(mask)
        short_x = self.cross_attn(short_x, x, key_padding_mask=key_padding_mask)
        short_x = short_x + self.ffn(self.ffn_norm(short_x))
        return short_x, short_mask


class SwiGLUEncoder(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, ffn_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + self.ffn(self.norm(x)), ensure_non_empty_mask(mask)


class SequenceRepresentationEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        short_seq_len: int,
    ) -> None:
        super().__init__()
        if encoder_type not in VALID_ENCODER_TYPES:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if encoder_type == "longer":
            self.impl = LongerStyleEncoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, short_seq_len=short_seq_len)
        elif encoder_type == "full_transformer":
            self.impl = FullTransformerEncoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden)
        else:
            self.impl = SwiGLUEncoder(d_model=d_model, ffn_hidden=ffn_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.impl(x, mask)


class QueryBoostMixer(nn.Module):
    def __init__(self, d_model: int, total_tokens: int, ffn_hidden: int) -> None:
        super().__init__()
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        self.total_tokens = total_tokens
        self.mixer_dim = math.ceil(d_model / total_tokens) * total_tokens
        self.in_proj = nn.Identity() if self.mixer_dim == d_model else nn.Linear(d_model, self.mixer_dim)
        self.out_proj = nn.Identity() if self.mixer_dim == d_model else nn.Linear(self.mixer_dim, d_model)
        self.token_norm = nn.LayerNorm(self.mixer_dim)
        self.channel_norm = nn.LayerNorm(self.mixer_dim)
        hidden_dim = max(ffn_hidden, self.mixer_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.mixer_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.mixer_dim),
        )

        sub_dim = self.mixer_dim // total_tokens
        sub_input_dim = total_tokens * sub_dim   # = mixer_dim
        sub_hidden_dim = sub_input_dim * 2
        self.sub_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sub_input_dim, sub_hidden_dim),
                nn.SiLU(),
                nn.Linear(sub_hidden_dim, sub_input_dim),
            )
            for _ in range(total_tokens)
        ])

    def token_mix(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, total_tokens, mixer_dim = x.shape
        if total_tokens != self.total_tokens:
            raise ValueError(f"Expected {self.total_tokens} tokens, but received {total_tokens}")
        sub_dim = mixer_dim // total_tokens


        x_grouped = x.view(batch_size, total_tokens, total_tokens, sub_dim).transpose(1, 2)
  

        x_mixed = torch.stack([
            mlp(x_grouped[:, i].reshape(batch_size, total_tokens * sub_dim))
            for i, mlp in enumerate(self.sub_mlps)
        ], dim=1)
  
        x_mixed = x_mixed.view(batch_size, total_tokens, total_tokens, sub_dim).transpose(1, 2)

        return x_mixed.reshape(batch_size, total_tokens, mixer_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.in_proj(x)
        x = x + self.token_mix(self.token_norm(x))

        x = x + self.ffn(self.channel_norm(x))
        return residual + self.out_proj(x)


class HyFormerLayer(nn.Module):
    def __init__(
        self,
        num_sequences: int,
        num_queries_per_sequence: int,
        num_non_seq_tokens: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        encoder_type: str,
        short_seq_len: int,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries_per_sequence = num_queries_per_sequence
        self.total_query_tokens = num_sequences * num_queries_per_sequence
        self.sequence_encoders = nn.ModuleList(
            [
                SequenceRepresentationEncoder(
                    encoder_type=encoder_type,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    short_seq_len=short_seq_len,
                )
                for _ in range(num_sequences)
            ]
        )
        self.query_decoders = nn.ModuleList(
            [CrossAttentionBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_sequences)]
        )
        self.query_boost = QueryBoostMixer(
            d_model=d_model,
            total_tokens=self.total_query_tokens + num_non_seq_tokens,
            ffn_hidden=ffn_hidden,
        )

    def forward(
        self,
        query_tokens: list[torch.Tensor],
        non_seq_tokens: torch.Tensor,
        sequence_tokens: list[torch.Tensor],
        sequence_masks: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        if len(query_tokens) != self.num_sequences:
            raise ValueError(f"Expected {self.num_sequences} query token groups, but received {len(query_tokens)}")

        decoded_queries: list[torch.Tensor] = []
        encoded_sequences: list[torch.Tensor] = []
        encoded_masks: list[torch.Tensor] = []

        for seq_idx in range(self.num_sequences):
            encoded_seq, encoded_mask = self.sequence_encoders[seq_idx](sequence_tokens[seq_idx], sequence_masks[seq_idx])
            decoded_query = self.query_decoders[seq_idx](
                query_tokens[seq_idx],
                encoded_seq,
                key_padding_mask=~ensure_non_empty_mask(encoded_mask),
            )
            encoded_sequences.append(encoded_seq)
            encoded_masks.append(encoded_mask)
            decoded_queries.append(decoded_query)

        mixed_tokens = torch.cat(decoded_queries + [non_seq_tokens], dim=1)
        boosted_tokens = self.query_boost(mixed_tokens)

        updated_queries: list[torch.Tensor] = []
        cursor = 0
        for _ in range(self.num_sequences):
            next_cursor = cursor + self.num_queries_per_sequence
            updated_queries.append(boosted_tokens[:, cursor:next_cursor, :])
            cursor = next_cursor
        updated_non_seq_tokens = boosted_tokens[:, cursor:, :]

        return updated_queries, updated_non_seq_tokens, encoded_sequences, encoded_masks, boosted_tokens


class HyFormerBackbone(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_sequences: int,
        num_queries_per_sequence: int,
        num_non_seq_tokens: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        encoder_type: str = "longer",
        short_seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HyFormerLayer(
                    num_sequences=num_sequences,
                    num_queries_per_sequence=num_queries_per_sequence,
                    num_non_seq_tokens=num_non_seq_tokens,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    encoder_type=encoder_type,
                    short_seq_len=short_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query_tokens: list[torch.Tensor],
        non_seq_tokens: torch.Tensor,
        sequence_tokens: list[torch.Tensor],
        sequence_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        current_seq_tokens = sequence_tokens
        current_seq_masks = sequence_masks
        boosted_tokens = torch.cat(query_tokens + [non_seq_tokens], dim=1)
        for layer in self.layers:
            query_tokens, non_seq_tokens, encoded_sequences, encoded_masks, boosted_tokens = layer(
                query_tokens, non_seq_tokens, current_seq_tokens, current_seq_masks
            )
            current_seq_tokens = encoded_sequences
            current_seq_masks = encoded_masks
        return boosted_tokens


def main() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_sequences = 3
    seq_len = 12
    seq_feature_dim = 6
    non_seq_dim = 32
    d_model = 16
    num_queries_per_sequence = 1
    num_non_seq_tokens = 5

    non_seq_x = torch.randn(batch_size, non_seq_dim)
    seq_x = torch.randn(batch_size, num_sequences, seq_len, seq_feature_dim)
    seq_masks = [(seq_x[:, idx].abs().sum(dim=-1) > 0) for idx in range(num_sequences)]
    sequence_tokens = [nn.Linear(seq_feature_dim, d_model)(seq_x[:, idx]) for idx in range(num_sequences)]
    non_seq_tokens = nn.Linear(non_seq_dim, num_non_seq_tokens * d_model)(non_seq_x).view(
        batch_size,
        num_non_seq_tokens,
        d_model,
    )

    pooled_sequences = [masked_mean_pool(sequence_tokens[idx], seq_masks[idx]) for idx in range(num_sequences)]
    global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
    query_generators = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(global_info.size(1), 32), nn.SiLU(), nn.Linear(32, num_queries_per_sequence * d_model))
            for _ in range(num_sequences)
        ]
    )
    query_tokens = [
        generator(global_info).view(batch_size, num_queries_per_sequence, d_model)
        for generator in query_generators
    ]

    backbone = HyFormerBackbone(
        num_layers=3,
        num_sequences=num_sequences,
        num_queries_per_sequence=num_queries_per_sequence,
        num_non_seq_tokens=num_non_seq_tokens,
        d_model=d_model,
        num_heads=4,
        ffn_hidden=32,
        encoder_type="longer",
        short_seq_len=4,
    )
    output_tokens = backbone(query_tokens, non_seq_tokens, sequence_tokens, seq_masks)

    print("Non-sequence feature [batch_size, non_seq_dim]:", tuple(non_seq_x.shape))
    print("Sequence feature [batch_size, num_sequences, seq_len, seq_feature_dim]:", tuple(seq_x.shape))
    print("Single query token group [batch_size, num_queries_per_sequence, d_model]:", tuple(query_tokens[0].shape))
    print("Non-sequence tokens [batch_size, num_non_seq_tokens, d_model]:", tuple(non_seq_tokens.shape))
    print("Single sequence token [batch_size, seq_len, d_model]:", tuple(sequence_tokens[0].shape))
    print("Output tokens [batch_size, num_sequences * num_queries_per_sequence + num_non_seq_tokens, d_model]:", tuple(output_tokens.shape))


if __name__ == "__main__":
    main()
