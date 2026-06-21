# Production Dataset Scripts Guide

这份文档解释当前新增的生产数据集脚本。目标是让你到公司后能快速确认三件事：

1. 训练标签到底应该用哪个字段。
2. 预处理和训练是否能完整跑通。
3. 如果要继续优化，应该先看哪些代码位置和 metadata 字段。

源码文件本身仍然保持英文和 ASCII；这份 Markdown 是给人看的说明文档。

## 一句话总览

当前生产数据流程分两步：

```text
production parquet
  -> scripts/preprocess_production.py
  -> data/production_sample/*.pt + metadata.json
  -> scripts/run_production.py
  -> HyFormer training + outputs/production/run_metadata.json
```

可以把它理解成：

- `preprocess_production.py` 负责把生产 parquet 翻译成模型能吃的张量。
- `run_production.py` 负责读取这些张量并训练 HyFormer。
- `utils/production_data.py` 是真正做字段识别、特征分组、标签生成和张量化的核心工具。

## 和 HyFormer 论文的对应关系

论文里的关键思想是：

- 非序列特征不要直接揉成一个大向量，而是按语义分组成多个 non-sequence tokens。
- 多种行为序列不要强行合并成一条大序列，而是每种序列独立建模。
- Query Generation 使用 non-sequence tokens 加序列池化摘要生成 query。
- 每层 HyFormer 先做 Query Decoding，再做 Query Boosting。

当前代码对应如下：

| 论文概念 | 当前实现 |
| --- | --- |
| Non-sequence tokens | `metadata["token_groups"]` 中的 user/context/query/candidate/cross/history/misc token |
| Multiple behavior sequences | `metadata["sequence_names"]` 中的 search/click/view/cart/buy/impression/item_hit 分支 |
| Query Generation | `models/taac_hyformer.py` 中 `build_query_tokens()` |
| Query Decoding | `main_pytorch.py` 中 `HyFormerLayer.query_decoders` |
| Query Boosting | `main_pytorch.py` 中 `QueryBoostMixer` |

最重要的一点：生产样本里已经有很多行为序列字段，所以不再像淘宝广告脚本那样自己从用户历史重建点击/曝光序列。

## 主要文件

### `utils/production_data.py`

这是生产数据处理的核心文件。

它负责：

- 读取 parquet。
- 自动分析每列是不是空、是不是列表、最大长度是多少。
- 根据字段名判断某列是非序列特征还是序列特征。
- 根据字段名判断某列是稀疏 ID 类特征还是连续数值类特征。
- 给稀疏 ID 做 bucket 编码。
- 给连续值做 `signed_log1p` 缩放。
- 生成 `non_seq_sparse.pt`、`non_seq_dense.pt`、`seq_sparse.pt`、`seq_dense.pt`、`seq_mask.pt`、`labels.pt` 对应的数据。
- 生成 `metadata.json`，把所有自动判断结果保存下来。

几个关键函数：

| 函数 | 作用 |
| --- | --- |
| `load_parquet_columns()` | 读取 parquet，并返回每列数据和 Arrow 类型 |
| `infer_column_infos()` | 统计每列的非空行数、最大长度、平均长度 |
| `sequence_branch_for_column()` | 根据字段名把列分到 search/click/view/cart/buy/impression/item_hit 序列 |
| `is_dense_name()` | 根据字段名判断是 dense 还是 sparse |
| `token_group_for_feature()` | 把非序列字段分到 user/context/query/candidate/cross/history/misc token |
| `label_from_columns()` | 根据 `--label-mode` 生成二分类标签 |
| `build_tensors()` | 总入口，把 parquet columns 转成全部训练张量和 metadata |

### `scripts/preprocess_production.py`

这是预处理入口脚本。

默认命令：

```bash
python scripts/preprocess_production.py --input-parquet 000000_0_selected_head100.parquet --output-dir data/production_sample --seq-len 100
```

它会输出：

```text
data/production_sample/
  labels.pt
  metadata.json
  non_seq_dense.pt
  non_seq_sparse.pt
  seq_dense.pt
  seq_mask.pt
  seq_sparse.pt
```

### `scripts/run_production.py`

这是训练入口脚本。

默认命令：

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 5 --batch-size 256 --save-checkpoint
```

它会：

- 读取预处理好的 `.pt` 文件。
- 根据 `metadata.json` 自动构建 `TAACHyFormerClassifier`。
- 按 `--val-ratio` 划分训练集和验证集。
- 使用 `BCEWithLogitsLoss` 训练二分类 CTR/相关性目标。
- 记录 AUC、logloss、accuracy。
- 写出 `outputs/production/run_metadata.json`。
- 如果加 `--save-checkpoint`，保存带时间戳的 checkpoint。

## 当前默认标签逻辑

现在默认使用：

```bash
--label-mode rel_score_present
```

含义是：

```text
rel_score_bkt >= 0 -> label 1
rel_score_bkt missing or -1 -> label 0
```

在 100 条样本里，这个规则得到：

```text
positive_samples = 34
negative_samples = 66
pos_rate = 0.34
```

这只是一个可训练的初始假设，不一定是最终正确标签。你到公司后最应该确认的就是这一点。

当前支持的标签模式：

| 参数 | 含义 |
| --- | --- |
| `rel_score_present` | `rel_score_bkt >= 0` 为正样本，默认值 |
| `rel_score_positive` | `rel_score_bkt > 0` 为正样本 |
| `rel_level_present` | `rel_level` 非空为正样本 |
| `rel_level_positive` | `rel_level > 0` 为正样本 |

如果公司确认真正训练目标是 `rel_level > 0`，预处理命令改成：

```bash
python scripts/preprocess_production.py --input-parquet YOUR_DATA.parquet --output-dir data/production_sample --seq-len 100 --label-mode rel_level_positive
```

## 特征如何被分组

### 第一步：判断是不是序列字段

代码看每列是不是列表型，以及字段名是否像行为历史。

会被当成序列的典型字段：

| 序列分支 | 字段名前缀或关键词 |
| --- | --- |
| `search_seq` | `last_query*`, `sess_q2q*`, `log_query*` |
| `click_seq` | `list_clk*`, `ups_clk*`, `ups_clkv2*`, 包含 `_clk` |
| `view_seq` | `last_view*`, `ups_view*`, 包含 `_view` |
| `cart_seq` | `ups_cart*`, 包含 `cart` |
| `buy_seq` | `ups_buy*`, 包含 `buy` |
| `impression_seq` | 包含 `impr`, `pagesn*`, `cur_pagesn*` |
| `item_hit_seq` | `i2i*`, 包含 `_hit_`, 以 `_hit_val` 结尾 |

这样做的原因很简单：生产数据已经提前把用户行为、召回命中、曝光、点击、浏览等信息放在字段里了，我们不要再自己伪造序列。

### 第二步：判断 sparse / dense

字段名像 ID、hash、category 的，默认当 sparse：

```text
id, ids, hash, goods_id, mall_id, cat1_id, cat2_id, cat3_id, cat4_id, user_id
```

字段名像统计值、分桶值、分数、长度、rank 的，默认当 dense：

```text
cnt, count, num, rate, score, ctr, cvr, sales, timediff, rank, idx, level, bkt, len, size, sim, val
```

sparse 特征会进 embedding，dense 特征会直接作为连续值输入 MLP。

### 第三步：非序列 token 分组

非序列字段会被分成这些 token group：

| Token group | 大概含义 |
| --- | --- |
| `user_token` | 用户 ID |
| `context_token` | 场景、站点、语言、页面、时区等上下文 |
| `query_token` | query、query term、query hash 等 |
| `candidate_token` | 当前候选商品、店铺、类目等 |
| `cross_token` | u2i、u2c、q2i、site_goods、target 等交叉统计 |
| `history_summary_token` | 行为序列的统计摘要 |
| `misc_token` | 暂时没有明确归类的字段 |

这些 token group 会传给 `TAACHyFormerClassifier`，每个 group 由一个 `SemanticTokenBuilder` 变成一个 non-sequence token。

## 序列特征如何变成模型输入

预处理输出两个序列张量：

```text
seq_sparse: [num_samples, num_sequences, seq_len, num_seq_sparse_fields]
seq_dense:  [num_samples, num_sequences, seq_len, num_seq_dense_fields]
seq_mask:   [num_samples, num_sequences, seq_len]
```

含义是：

- 第一维是样本数。
- 第二维是序列分支数，比如 search/click/view 等。
- 第三维是序列长度。
- 第四维是每个时间步携带的字段。

`seq_mask` 表示某个位置是否真的有行为。没有行为的位置是 padding。

默认：

```bash
--sequence-truncation tail
```

也就是序列太长时保留最后 `seq_len` 个值。这个更像“保留最近行为”。如果公司数据的列表本来就是按重要性排序而不是时间排序，可以考虑改成：

```bash
--sequence-truncation head
```

## 非序列特征如何变成模型输入

预处理输出两个非序列张量：

```text
non_seq_sparse: [num_samples, num_non_seq_sparse_fields]
non_seq_dense:  [num_samples, num_non_seq_dense_fields]
```

对于非序列 sparse 字段：

- 取第一个标量值。
- 做 bucket 编码。
- 0 保留给 padding/missing。

对于非序列 dense 字段：

- 标量字段直接做 `signed_log1p`。
- 列表字段会生成 6 个摘要统计：

```text
length_log, mean, std, min, max, last
```

这也是为什么 `non_seq_dense` 会比较宽。100 条样本里当前是：

```text
non_seq_sparse: (100, 35)
non_seq_dense:  (100, 681)
```

## 100 条样本当前预处理结果

在本地小样本上，预处理已经通过，结果是：

```text
labels:         (100,)
non_seq_sparse: (100, 35)
non_seq_dense:  (100, 681)
seq_sparse:     (100, 7, 64, 74)
seq_dense:      (100, 7, 64, 17)
seq_mask:       (100, 7, 64)
```

当前 7 条序列是：

```text
search_seq
click_seq
view_seq
cart_seq
buy_seq
impression_seq
item_hit_seq
```

注意：这个验证只证明预处理能跑通。由于本机内存小，没有继续跑训练。

## 公司机器上建议先跑的命令

### 1. 先跑小样本预处理

```bash
python scripts/preprocess_production.py --input-parquet 000000_0_selected_head100.parquet --output-dir data/production_sample --seq-len 100
```

看输出里这些字段：

```text
labels
pos_rate
non_seq_sparse
non_seq_dense
seq_sparse
seq_dense
seq_mask
sequence_names
token_groups
```

### 2. 如果特征太宽，先限制每条序列的字段数

```bash
python scripts/preprocess_production.py --input-parquet 000000_0_selected_head100.parquet --output-dir data/production_sample --seq-len 100 --max-seq-fields-per-branch 16
```

这个参数会在每个序列分支里优先保留覆盖样本多、平均长度长的字段。

### 3. 再跑一小轮训练

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 128 --d-model 64 --ffn-hidden 128 --hyformer-layers 2 --save-checkpoint
```

如果显存够，可以加：

```bash
--amp
```

## 公司机器上要重点检查的 metadata

预处理后先打开：

```text
data/production_sample/metadata.json
```

重点看这些字段：

### `label_mode`

确认当前标签模式是不是你想要的。

```json
"label_mode": "rel_score_present"
```

### `pos_rate`

正样本比例是否合理。

如果正样本比例离业务常识很远，优先怀疑标签模式不对。

### `sequence_names`

确认当前识别到哪些序列。

```json
"sequence_names": [
  "search_seq",
  "click_seq",
  "view_seq",
  "cart_seq",
  "buy_seq",
  "impression_seq",
  "item_hit_seq"
]
```

### `sequence_fields`

确认每条序列里放了哪些原始字段。

如果某些字段被放错分支，就改 `sequence_branch_for_column()`。

### `sequence_non_empty_samples`

看每条序列有多少样本非空。

如果某条序列几乎全空，可以考虑先去掉，减少模型噪音和内存。

### `token_groups`

确认非序列 token 的语义分组是否合理。

如果某些核心字段落入 `misc_token`，后续应该把它们明确分到 query/candidate/cross/history 等更合适的 token。

### `column_overview`

这是每列的结构统计。

重点看：

- `non_empty_rows`
- `max_flat_len`
- `mean_flat_len`
- `scalar_like`

这些能帮助判断字段是不是序列、是不是几乎全空、是不是异常长。

## 常见问题和判断方式

### 问题 1：AUC 训练出来很奇怪

优先检查标签。

建议按顺序试：

```bash
--label-mode rel_score_present
--label-mode rel_score_positive
--label-mode rel_level_present
--label-mode rel_level_positive
```

然后比较：

- `pos_rate`
- train AUC
- val AUC
- logloss

如果某个标签模式下 `pos_rate` 明显更符合业务，优先用那个。

### 问题 2：预处理内存很高

先降低：

```bash
--seq-len 50
--max-seq-fields-per-branch 16
```

如果全量数据很大，后续应该做分块 parquet 预处理，而不是一次把所有列全部读入内存。当前脚本是为了先打通生产样本路径，还不是最终大规模离线预处理器。

### 问题 3：训练内存很高

先降低：

```bash
--batch-size 64
--d-model 32
--ffn-hidden 64
--hyformer-layers 1
--short-seq-len 8
```

然后逐步加大。

### 问题 4：某些序列分组不合理

改这里：

```text
utils/production_data.py
  sequence_branch_for_column()
```

比如公司确认 `last_query_impr_goods_ids` 实际是曝光列表，而不是搜索序列，就应该把它从 `search_seq` 规则挪到 `impression_seq`。

### 问题 5：某些非序列 token 分组不合理

改这里：

```text
utils/production_data.py
  token_group_for_feature()
```

比如公司确认某些 `target_*` 字段是候选商品强相关特征，而不是 cross feature，可以把规则调到 `candidate_token`。

## 后续最值得优化的方向

### 1. 标签最终确认

这是第一优先级。

当前 `rel_score_bkt` 是为了让链路先跑起来的假设。真正上线训练前，需要确认：

- 哪一列是曝光/点击/转化标签。
- 是否有 sample weight。
- 是否需要 query-level AUC。
- 是否存在一行多候选或一行单候选的语义差异。

### 2. 手工修正序列分组

当前分组是根据字段名启发式判断。

更好的版本应该由业务同学或特征字典确认：

- 哪些字段是搜索行为。
- 哪些字段是浏览行为。
- 哪些字段是点击行为。
- 哪些字段是购买/加购行为。
- 哪些字段只是统计特征，不应该作为逐步序列输入。

### 3. 减少过宽序列 side features

当前会把很多序列字段都塞进对应分支。

后续可以做：

- 按覆盖率筛字段。
- 按字段重要性筛字段。
- 对超长 ID 序列做 top-k。
- 对统计型序列只保留核心 rank/count/time_diff。

### 4. 改成大数据友好的分块预处理

当前脚本为了快速打通样本，是一次读整个 parquet。

全量生产数据更合理的方式是：

```text
read parquet by batch
  -> vectorize batch
  -> save shard tensors
  -> train with streaming/sharded loader
```

这会明显降低内存压力。

### 5. 指标改成业务一致版本

当前训练脚本是普通 sample-level AUC。

如果生产基线使用 query-level AUC，需要在 `run_production.py` 里读取 query/request id，并按 query 分组计算 AUC。

## 当前结论

目前代码已经完成的是“能把生产 parquet 样本翻译成 HyFormer 输入”的第一版。

这版的价值是：

- 不再伪造序列，直接使用生产特征里的行为序列。
- 多种行为序列分开建模，符合 HyFormer 论文里的 multi-sequence 思路。
- 非序列特征按语义分 token，贴近论文的 semantic grouping。
- Query Generation 继续使用 non-sequence tokens 加 sequence pooling summaries。

下一步最关键不是马上改模型，而是先在公司确认标签和字段语义。标签确认后，再根据 metadata 里的分组结果决定哪些字段该保留、移动或删除。
