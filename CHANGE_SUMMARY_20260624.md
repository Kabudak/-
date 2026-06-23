# 2026-06-24 生产数据特征加工改造总结

本文档总结当前这一轮为了适配生产数据而做的全部改动。重点是：重新控制特征数量、重选序列骨干字段、让 NS-token 和 sequence-token 的构造方式更接近 HyFormer 统一异构特征的思想。

## 一、改造背景

上一版生产数据预处理可以跑通，但 review 后发现两个主要问题。

第一，NS 部分有很多数组特征。上一版对 sparse 数组做 embedding 后 pooling，这个方向还比较合理；但对 dense 数组做多种统计摘要，会把一个原始特征展开成多个内部特征，导致特征数量膨胀，影响训练和推理效率。

第二，Sequence 部分原本想表达用户历史行为序列，但上一版字段比较杂，很多字段形状不同，直接拼接后语义不清晰，也不利于稳定构造每个行为 step。

因此本轮改造的核心原则是：

1. NS-token 优先使用单值特征。
2. NS 数组特征尽量压缩为一个字段级输入，不再大量展开。
3. 每条行为序列单独选择一个主字段作为骨干。
4. 其他序列字段只作为该骨干 step 的附加信息。
5. 每个特征先映射成固定长度向量，再分组拼接，通过共享 MLP 得到 token。

## 二、当前特征分组

特征文件：

```text
data/selectedfeaturefinal.txt
```

当前分组如下：

| 分组 | 模型角色 |
| --- | --- |
| `label` | 训练标签 |
| `context` | `context_token` |
| `item` | `item_token` |
| `click` | `click_seq` |
| `impr` | `impression_seq` |
| `buy` | `buy_seq` |

当前标签字段：

```text
label_click
```

## 三、NS-token 改动

NS 部分包括：

```text
context_token
item_token
```

### 1. Context 特征

当前 context 特征包括站点、场景、语言、地区、页面、query、部分用户历史摘要等。

示例：

```text
currency
language
page_elsn
page_sn
plat
region
scene_id
site_id
timezone
search_method
opt_id
req_goods_id
origin_query_hash
query_hash
ups_query_term_hash_v2
flip_cat1_ids
flip_mall_ids
list_clk_cat1_ids
log_all_impr_tg_1d
log_all_view_tg
```

### 2. Item 特征

当前 item 特征优先选候选商品自身属性、类目、商家、价格、销量、召回关系和少量统计特征。

示例：

```text
goods_id
mall_id
cat1_id
cat2_id
cat3_id
cat4_id
sellr_type
rel_level
adj_ctr
adj_cvr
adj_cartcvr
create_time
price
sales
auto_price_mul10_dis
u2i_impr_clk_30d_lg
price_div_20
price_div_50
mkt_prc_div_200
cart_cnt_3d
flip_cat_cnt
u2i_ctr_30d
query_goods_clk_size_crs_hit_ctr
multimodal_i2i_hit_cart_size
multimodal_i2i_hit_clk_size
goods_name_bigram_hash
```

### 3. NS 数组处理方式

本轮不再把 dense 数组展开成多种统计摘要。

当前策略：

| 类型 | 处理方式 |
| --- | --- |
| sparse 标量 | 取最近一个值，bucket 后 embedding |
| dense 标量或 dense 数组 | 取最近一个值，必要时做 signed log1p |
| sparse 数组 | 保留 bag，bucket 后 embedding，再 masked mean pooling |

当前 sparse bag 字段包括：

```text
flip_cat1_ids
flip_mall_ids
list_clk_cat1_ids
origin_query_hash
query_hash
ups_query_term_hash_v2
goods_name_bigram_hash
```

这样做的目的：每个原始字段最多贡献一个字段级向量，避免 dense 摘要把特征数量放大。

## 四、Sequence 改动

本轮明确了一个关键约定：

每条行为序列都要有自己的骨干字段，mask 只由骨干字段决定。

附加字段只拼接到该 step 上，不参与判断该 step 是否有效。

### 1. Click 序列

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

默认长度：

```text
100
```

### 2. Impression 序列

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

默认长度：

```text
200
```

说明：上一版使用 `*_l10` 字段，长度太短；当前已切换到更长的 `ups_view_*` 系列。

### 3. Buy 序列

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

默认长度：

```text
200
```

说明：当前 parquet 中没有找到能和 `ups_buy_ids` 完整对齐的 200 长度 cat1、cat2、cat3、cat4 分层字段，所以先使用能和骨干字段对齐的 `ups_buy_cat_ids_x`。

## 五、序列长度配置

三条序列长度可以不同。

当前推荐配置：

```bash
--seq-len 200 --sequence-lens click_seq=100,impression_seq=200,buy_seq=200
```

内部 tensor 会 pad 到最大长度，也就是 200。较短分支超过自身配置长度的位置 mask 为 False。

## 六、模型结构改动

文件：

```text
models/taac_hyformer.py
```

本轮新增了两个共享编码器：

```text
NonSequenceTokenEncoder
SharedSequenceStepEncoder
```

### 1. NS-token 编码

每个 NS 字段先映射成长度为 `field_embed_dim` 的向量。

默认：

```text
field_embed_dim = 64
```

同一个 token group 内部的字段向量拼接后，进入共享 MLP：

```text
num_fields * 64 -> 320 -> 128
```

输出就是一个 NS-token。

默认：

```text
token_mlp_hidden = 320
d_model = 128
```

当前 `context_token` 和 `item_token` 共用这套 token encoder。

### 2. Sequence step 编码

每个序列 step 中的字段也先映射成长度为 64 的向量，然后拼接，进入共享 MLP：

```text
num_step_fields * 64 -> 320 -> 128
```

输出就是这个 step 的 sequence-token。

当前 click、impression、buy 三条序列共享一个 step encoder。后续可以考虑改成每个序列分支独立 encoder。

## 七、预处理代码改动

文件：

```text
utils/production_data.py
scripts/preprocess_production.py
```

主要变化：

1. 支持每个 sequence branch 独立长度。
2. 记录 `sequence_lens` 到 metadata。
3. 记录 `sequence_backbone_fields` 到 metadata。
4. 序列 mask 只由各分支骨干字段生成。
5. dense 数组不再展开成 `mean/std/min/max/last` 等摘要特征。
6. sparse bag 仍保留 masked mean pooling 路线。
7. 增加显式 dense 字段集合，避免价格、时间、log、tg 类字段被误判成 sparse ID。

新增或重要参数：

```text
--sequence-lens
--non-seq-array-reduction
--non-seq-bag-len
```

## 八、训练脚本改动

文件：

```text
scripts/run_production.py
```

主要变化：

1. 将 `field_embed_dim` 默认改为 64。
2. 将 `d_model` 默认改为 128。
3. 新增 `--token-mlp-hidden`，默认 320。
4. 从 metadata 读取 `sequence_fields` 和 `sequence_names`，传给模型。
5. 继续兼容已有 sparse bag 输入。

推荐 smoke training 命令：

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 128 --field-embed-dim 64 --token-mlp-hidden 320 --ffn-hidden 256 --hyformer-layers 1 --short-seq-len 8
```

## 九、命令文档改动

文件：

```text
Test.md
```

当前推荐预处理命令：

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0.gz_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 200 --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 --non-seq-bag-len 64 --non-seq-array-reduction last
```

当前推荐训练命令：

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 128 --field-embed-dim 64 --token-mlp-hidden 320 --ffn-hidden 256 --hyformer-layers 1 --short-seq-len 8
```

## 十、验证结果

本机没有跑完整训练，只做轻量验证。

已通过：

```text
py_compile
2-row preprocess
tiny model forward
code/config non-ascii scan
git diff --check
```

最近一次 2 行检查命令：

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0.gz_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample_check --max-rows 2 --seq-len 12 --sequence-lens click_seq=8,impression_seq=12,buy_seq=12 --non-seq-bag-len 16
```

输出形状：

```text
non_seq_sparse:   (2, 19)
non_seq_bag:      (2, 7, 16)
non_seq_bag_mask: (2, 7, 16)
non_seq_dense:    (2, 20)
seq_sparse:       (2, 3, 12, 16)
seq_dense:        (2, 3, 12, 3)
seq_mask:         (2, 3, 12)
```

小模型前向：

```text
logits_shape (2, 1)
finite True
```

mask 检查：

```text
[[0, 0, 0], [8, 12, 0]]
```

含义：第二条样本 click 有 8 个有效 step，impression 有 12 个有效 step，buy 没有购买历史。

## 十一、当前改动文件清单

主要改动文件：

```text
data/selectedfeaturefinal.txt
utils/production_data.py
models/taac_hyformer.py
scripts/preprocess_production.py
scripts/run_production.py
Test.md
PRODUCTION_PIPELINE.md
```

新增本文档：

```text
CHANGE_SUMMARY_20260624.md
```

验证过程会更新：

```text
data/production_sample_check/metadata.json
```

如果不想提交验证产物，可以在提交前单独还原或取消暂存该文件。

## 十二、后续建议

1. 在公司机器上先跑完整预处理，确认完整数据中三条序列字段长度是否稳定。
2. 检查 `ups_buy_cat_ids_x` 是否足够表达 buy 类目信息，后续如果找到 cat1-4 长序列可替换。
3. 观察 `impression_seq=200`、`buy_seq=200` 对显存和速度的影响。
4. 如果内存压力大，优先降低 batch size，其次再考虑降低序列长度。
5. 后续实验可以尝试 context 和 item 使用 specific MLP，而不是共享 MLP。
6. 后续实验可以尝试 click、impression、buy 三条序列使用独立 step encoder。
