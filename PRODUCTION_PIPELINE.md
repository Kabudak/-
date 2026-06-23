# 生产数据 HyFormer 特征加工说明

这份文档说明当前生产数据版本的特征选择、预处理方式和模型输入结构。

当前目标不是把所有可用特征都塞进模型，而是先保证输入结构符合 HyFormer 统一处理异构特征的思路，并控制特征数量，方便后续在公司机器上做稳定实验。

## 一、核心设计

当前样本被组织成两类输入：

1. NS-token：非序列特征，例如 context 和 item。
2. Sequence-token：用户历史行为序列，例如 click、impression、buy。

当前 NS-token 有两个：

```text
context_token
item_token
```

当前 sequence branch 有三个：

```text
click_seq
impression_seq
buy_seq
```

三条序列不是共用一个骨干字段。每条序列都有自己的主特征作为骨干：

| 分支 | 骨干字段 | 含义 |
| --- | --- | --- |
| `click_seq` | `ups_clkv2_7d_ids` | 用户点击过的商品 ID 序列 |
| `impression_seq` | `ups_view_goods_ids` | 用户曝光或浏览过的商品 ID 长序列 |
| `buy_seq` | `ups_buy_ids` | 用户购买过的商品 ID 长序列 |

每条序列的 mask 只由自己的骨干字段决定。附加字段只是拼接到同一个 step 上，不会反过来决定这个 step 是否有效。

这样可以避免一种常见问题：某些附加字段有占位值，但主行为其实为空。如果用所有字段共同生成 mask，就可能把空行为误判成有效行为。

## 二、当前人工特征文件

特征分组文件是：

```text
data/selectedfeaturefinal.txt
```

当前分组：

| 分组 | 角色 |
| --- | --- |
| `label` | 标签字段 |
| `context` | 上下文 NS-token |
| `item` | 候选商品 NS-token |
| `click` | 点击序列 |
| `impr` | 曝光或浏览序列 |
| `buy` | 购买序列 |

当前标签字段：

```text
label_click
```

## 三、NS-token 特征处理

NS 部分包含 `context` 和 `item`。

当前原则是：优先选单值字段。对于数组字段，不再把 dense 数组展开成多个摘要统计值，而是尽量压缩成一个字段级输入，避免特征数量膨胀。

## 1. Sparse 标量字段

ID 类字段会进入 sparse 通道。

处理方式：

1. 对原始值取最近一个值，也就是数组 flatten 后的最后一个值。
2. 转成整数。
3. 进入字段级 bucket。
4. 查 embedding。

例如：

```text
site_id
page_sn
goods_id
mall_id
cat1_id
cat2_id
cat3_id
cat4_id
```

## 2. Dense 标量字段

连续数值字段会进入 dense 通道。

处理方式：

1. 对原始值取最近一个值。
2. 如果字段不是已 log 字段，则做 signed log1p。
3. 每个 dense 字段先映射成一个长度为 `field_embed_dim` 的向量。

当前显式固定走 dense 的字段包括：

```text
create_time
log_all_impr_tg_1d
log_all_view_tg
log_all_clk_tg
mkt_prc_div_200
price
price_div_20
price_div_50
u2i_impr_clk_30d_lg
ups_buy_tg_v2_l10
ups_buy_tg
ups_view_tg_v2_l10
ups_view_tg
```

## 3. Sparse 数组字段

Sparse 数组字段仍然保留 bag 形式。

处理方式：

1. flatten 原始数组。
2. 保留最多 `--non-seq-bag-len` 个值。
3. bucket 化每个 ID。
4. 保存 ID tensor 和 mask。
5. 模型里查 embedding 后做 masked mean pooling。
6. pooled vector 作为这个字段的一个向量输入。

这样做符合 HyFormer 对异构特征统一 token 化的思路，同时不会把 NS 数组扩成额外序列。

当前 NS sparse bag 字段包括：

```text
flip_cat1_ids
flip_mall_ids
list_clk_cat1_ids
origin_query_hash
query_hash
ups_query_term_hash_v2
goods_name_bigram_hash
```

## 四、NS-token 编码方式

当前模型里，每个原始特征都会先变成一个长度为 `field_embed_dim` 的向量。

默认生产参数：

```text
field_embed_dim = 64
d_model = 128
token_mlp_hidden = 320
```

一个 token group 内的所有字段向量会拼接起来。例如某组有 10 个字段：

```text
10 * 64 = 640
```

然后经过共享 MLP：

```text
640 -> 320 -> 128
```

输出就是一个长度为 128 的 NS-token。

当前 NS-token 侧的 MLP 是共享的。也就是说，`context_token` 和 `item_token` 使用同一个 token encoder 结构和同一套 MLP 权重。后续如果实验需要，可以再改成每个语义组独立 MLP。

## 五、Sequence 特征处理

当前三条序列分别独立选择骨干字段。

## 1. Click 序列

骨干字段：

```text
ups_clkv2_7d_ids
```

附加字段：

```text
ups_clkv2_7d_cat1_ids
ups_clkv2_7d_cat2_ids
ups_clkv2_7d_cat3_ids
ups_clkv2_7d_cat4_ids
ups_clkv2_7d_mall_ids
ups_clkv2_7d_page_sns
ups_clkv2_7d_page_elsns
```

这些字段在小样本中长度完全一致，适合作为同一条点击行为序列的 step 级附加信息。

## 2. Impression 序列

骨干字段：

```text
ups_view_goods_ids
```

附加字段：

```text
ups_view_cat1_ids
ups_view_cat_ids
ups_view_cat4_ids
ups_view_mall_ids
ups_view_page_sns
ups_view_tg
```

其中 `ups_view_tg` 被当作 dense 时间类特征。这组字段在样本中长度一致，最大长度可到 200。

## 3. Buy 序列

骨干字段：

```text
ups_buy_ids
```

附加字段：

```text
ups_buy_cat_ids_x
ups_buy_tg
ups_buy_prices_dis
```

其中 `ups_buy_tg` 和 `ups_buy_prices_dis` 走 dense 通道。当前 parquet 里没有完整 200 长度的 buy cat1-4 分层字段，所以先使用可以和 `ups_buy_ids` 对齐的 `ups_buy_cat_ids_x`。

## 六、三条序列长度可以不同

当前预处理支持每个分支独立配置序列长度：

```bash
--sequence-lens click_seq=100,impression_seq=200,buy_seq=200
```

内部 tensor 仍然会 pad 到最大长度，方便沿用原来的模型结构。

例如：

```text
click_seq       -> 100
impression_seq  -> 200
buy_seq         -> 200
```

最终 `seq_sparse` 和 `seq_dense` 的第三维会是三个分支长度的最大值。短分支之外的位置会 pad，mask 为 False。

## 七、Sequence-token 编码方式

序列中的每个 step 会按类似 NS-token 的方式编码。

每个 step 中的字段先各自变成长度为 `field_embed_dim` 的向量，然后拼接，再过共享 MLP：

```text
num_step_fields * 64 -> 320 -> 128
```

输出就是这个 step 的 sequence token，长度为 128。

三条序列共享同一个 sequence step encoder。后续如果需要，也可以改成每条序列独立 encoder。

## 八、当前输出文件

预处理会生成：

```text
non_seq_sparse.pt
non_seq_sparse_bag.pt
non_seq_sparse_bag_mask.pt
non_seq_dense.pt
seq_sparse.pt
seq_dense.pt
seq_mask.pt
labels.pt
metadata.json
```

其中 `metadata.json` 会记录：

- `sequence_names`
- `sequence_lens`
- `sequence_backbone_fields`
- `sequence_fields`
- `token_groups`
- `non_seq_sparse_fields`
- `non_seq_sparse_bag_fields`
- `non_seq_dense_fields`
- `seq_sparse_fields`
- `seq_dense_fields`

训练脚本会用这些信息自动构建模型输入结构。

## 九、推荐命令

预处理：

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0.gz_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 200 --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 --non-seq-bag-len 64 --non-seq-array-reduction last
```

小模型 smoke training：

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 128 --field-embed-dim 64 --token-mlp-hidden 320 --ffn-hidden 256 --hyformer-layers 1 --short-seq-len 8
```

如果显存或内存不足，优先降低：

1. `--batch-size`
2. `--seq-len`
3. `--non-seq-bag-len`
4. `--d-model`
5. `--field-embed-dim`

## 十、已完成的轻量验证

本机没有跑完整训练，只做轻量检查。

2 行预处理检查通过，关键形状为：

```text
non_seq_sparse:   (2, 19)
non_seq_bag:      (2, 7, 16)
non_seq_bag_mask: (2, 7, 16)
non_seq_dense:    (2, 20)
seq_sparse:       (2, 3, 12, 16)
seq_dense:        (2, 3, 12, 3)
seq_mask:         (2, 3, 12)
```

使用 `click_seq=8,impression_seq=12,buy_seq=12` 做小检查时：

```text
sequence_lens: {'click_seq': 8, 'impression_seq': 12, 'buy_seq': 12}
sequence_backbone_fields: {'click_seq': 'ups_clkv2_7d_ids', 'impression_seq': 'ups_view_goods_ids', 'buy_seq': 'ups_buy_ids'}
```

小模型前向检查也已通过：

```text
logits_shape (2, 1)
finite True
```

## 十一、后续可优化方向

1. 继续检查完整生产数据上三条序列字段长度是否稳定一致。
2. 评估 NS sparse bag 是否继续 mean pooling，还是改成 attention pooling。
3. 对 `context_token` 和 `item_token` 尝试 specific MLP。
4. 对 click、impression、buy 尝试各自独立 step encoder。
5. 检查高基数字段 bucket 冲突率。
