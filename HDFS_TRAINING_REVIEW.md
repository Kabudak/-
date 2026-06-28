# HDFS 流式训练排查与后续优化说明

## 1. 当前问题背景

生产训练数据现在通过 HDFS 读取，预处理、验证、训练都集成在 `production_hdfs_dataset.py` 和 `scripts/run_production_hdfs.py` 中，形成按 parquet batch 流式读取、流式预处理、渐进式训练的流程。

当前平台现象是：`batch_size=4096` 跑到约 100 个 batch 后，AUC 一直在 `0.49-0.51` 之间波动，基本等价于无效模型。这个现象可能来自三类问题：

1. 数据没有被正确解析进模型，例如数组字段、标签字段、序列字段被读成空值或 0。
2. 数据本身信号弱，例如标签列选择不合适、正负样本分布异常、特征缺失率过高。
3. 模型输入结构或模型容量不适配当前生产数据。

这次优先修复的是第一类问题，并补充了平台上排查第二类问题所需的 debug 输出和训练参数。

## 2. 已完成的代码改动

### 2.1 修复 numpy.ndarray 解析问题

从 Excel 样本可以看到，很多字段不是普通标量，而是类似下面的形态：

```text
[array([634418220433963])]
[array([601, 602])]
[25439 9711 25439 ...]
```

HDFS parquet 经过 `record_batch.to_pandas()` 后，嵌套数组字段很容易变成 `numpy.ndarray`。旧逻辑主要处理 `list` 和 `tuple`，没有完整处理 `numpy.ndarray`，会导致很多字段被解析为空或 0。

本次在 `utils/production_data.py` 中修复了以下函数：

- `flatten_values`
- `sequence_values`
- `first_scalar`
- `safe_float`
- `safe_int`

修复后，下面这些形态可以被正确解析：

```python
[np.array([1])] -> 1
[np.array([601, 602])] -> [601, 602]
```

这会影响所有依赖这些基础函数的路径，包括：

- 标签列 `label_click`
- NS sparse scalar
- NS sparse bag
- NS dense
- Sequence sparse
- Sequence dense

### 2.2 标签列改为严格检查

旧逻辑在标签列缺失时可能默默补 0，这在训练中非常危险，因为脚本会继续跑，但实际是在训练全负样本。

现在两条路径都改为严格检查：

- HDFS 流式路径：`production_hdfs_dataset.py`
- 离线 sample 路径：`utils/production_data.py`

如果缺少 `label_click`，会直接报错，而不是继续训练。

### 2.3 HDFS 多 worker 读取去重

`ProductionHDFSDataset` 是 `IterableDataset`。如果直接开启多个 DataLoader worker，但不做 worker sharding，多个 worker 可能会重复读取相同文件。

现在 `production_hdfs_dataset.py` 中加入了 `get_worker_info()`，每个 worker 只读取自己负责的一部分文件：

```text
worker 0: file_list[0::num_workers]
worker 1: file_list[1::num_workers]
...
```

这样可以安全开启 `--num-workers` 提升 CPU 预处理吞吐。

### 2.4 增加 DataLoader 性能参数

`scripts/run_production_hdfs.py` 新增：

- `--num-workers`
- `--prefetch-factor`
- `--pin-memory`

同时，batch tensor 搬到 GPU 时支持 `non_blocking=True`。在 CUDA 环境下，推荐配合：

```bash
--num-workers 2 --pin-memory --prefetch-factor 2
```

如果 CPU 仍然是瓶颈，再逐步尝试 `--num-workers 4` 或 `--num-workers 8`。

### 2.5 降低每 batch 评估带来的同步开销

旧训练循环几乎每个 batch 都会：

1. 额外做一次 eval forward。
2. 把 logits 从 GPU 拷回 CPU。
3. 计算 AUC、logloss、accuracy。

这会让 GPU 经常等待 CPU，吞吐会比较差。

现在新增：

- `--eval-every-batches`
- `--train-metrics-every`

默认值为：

```bash
--eval-every-batches 20 --train-metrics-every 20
```

含义是：

- 训练仍然每个 batch 都进行。
- pre-train eval 默认每 20 个 batch 做一次。
- train AUC/logloss 默认每 20 个 batch 算一次。
- 最后一批仍然保留 test-only 评估。

如果需要复现旧行为，可以设置：

```bash
--eval-every-batches 1 --train-metrics-every 1
```

### 2.6 修正 HDFS 默认序列长度

HDFS 脚本之前默认 `--seq-len 100`，这会导致 impression 和 buy 序列也默认被截到 100，和前面讨论的设计不一致。

现在默认改为：

```bash
--seq-len 200 --sequence-lens click_seq=100,impression_seq=200,buy_seq=200
```

含义是：

- click 序列最大长度 100
- impression 序列最大长度 200
- buy 序列最大长度 200
- 全局 tensor 最大长度为 200

### 2.7 增加 debug batch 数据健康输出

新增参数：

```bash
--debug-batches 5
```

会打印前 N 个 batch 的关键信息：

- `pos_rate`：正样本比例
- `labels`：当前 batch 中出现的标签值
- `ns_sparse_nz`：NS 稀疏标量非零率
- `ns_bag_mask`：NS sparse bag 有效位置比例
- `ns_dense_abs_mean`：NS dense 绝对值均值
- `seq_sparse_nz`：Sequence 稀疏特征非零率
- `seq_dense_abs_mean`：Sequence dense 绝对值均值
- `seq_mask_mean`：每条序列的平均有效长度
- `dense_finite`：dense 是否存在 NaN 或 Inf
- `logits_mean/logits_std`：模型输出均值和方差

这些指标比单纯看 AUC 更适合判断数据是否真的进了模型。


## 3. 已完成的本地验证

### 3.1 Python 编译检查

已通过：

```bash
python -m py_compile main_pytorch.py models/taac_hyformer.py production_hdfs_dataset.py scripts/run_production_hdfs.py utils/production_data.py
```

### 3.2 ndarray 解析检查

已验证：

```text
first 1
safe_int 1
safe_float 1.0
flatten [601, 602]
sequence [601, 602]
bucket [2]
label 1
```

### 3.3 diff 格式检查

已通过：

```bash
git diff --check
```

### 3.4 本地未完成项

本地 torch 环境缺少 `pyarrow`，因此没有运行 HDFS 脚本的 `--help` 和真实 HDFS 训练。这个不影响生产平台验证，因为平台上已经可以跑通 HDFS 脚本。

## 4. 明天平台第一轮建议命令

下面命令用于确认修复后的数据是否健康。请把 `hdfs://namenode:8020/path/to/day` 替换成真实的一天数据路径。

```bash
python scripts/run_production_hdfs.py \
  --data-path hdfs://namenode:8020/path/to/day \
  --feature-file data/selectedfeaturefinal.txt \
  --parquet-batch-size 4096 \
  --seq-len 200 \
  --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 \
  --non-seq-bag-len 64 \
  --non-seq-array-reduction last \
  --amp \
  --num-workers 2 \
  --pin-memory \
  --prefetch-factor 2 \
  --debug-batches 5 \
  --eval-every-batches 20 \
  --train-metrics-every 20 \
  --save-checkpoint
```

第一轮不要急着看最终 AUC，优先看 debug 输出是否正常。

重点检查：

1. `labels` 是否同时包含 0 和 1。
2. `pos_rate` 是否合理，不能长期为 0 或 1。
3. `seq_mask_mean` 是否不是全 0。
4. `ns_sparse_nz` 和 `seq_sparse_nz` 是否有明显非零值。
5. `dense_finite` 是否为 `True`。
6. `logits_std` 是否长期接近 0。
7. loss 是否随着 batch 推进有下降趋势。

如果 `logits_std` 长期接近 0，说明模型对不同样本输出几乎一样，通常表示特征没有有效进入模型，或者模型初期梯度很弱。

## 5. 测试数据信号的训练指令

### 5.1 小样本过拟合测试

目的：验证模型能不能记住很小的一批数据。

如果模型连很小的数据都记不住，AUC 仍然长期接近 0.5，那么问题大概率还在标签、特征解析、mask、loss 或模型输入链路。

建议准备一个只包含少量 parquet 文件的小目录，例如一天数据中的一个小时或更小分片，然后跑多 epoch：

```bash
python scripts/run_production_hdfs.py \
  --data-path hdfs://namenode:8020/path/to/small_slice \
  --feature-file data/selectedfeaturefinal.txt \
  --parquet-batch-size 4096 \
  --seq-len 200 \
  --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 \
  --non-seq-bag-len 64 \
  --non-seq-array-reduction last \
  --epochs 5 \
  --lr 1e-3 \
  --amp \
  --num-workers 2 \
  --pin-memory \
  --prefetch-factor 2 \
  --debug-batches 5 \
  --eval-every-batches 1 \
  --train-metrics-every 1
```

预期现象：

- train loss 应该明显下降。
- train AUC 应该逐步高于 0.5。
- 如果多轮后仍然完全不动，需要继续排查输入和标签。

### 5.2 低采样率快速烟测

如果暂时没有小目录，可以用文件采样快速跑一版：

```bash
python scripts/run_production_hdfs.py \
  --data-path hdfs://namenode:8020/path/to/day \
  --feature-file data/selectedfeaturefinal.txt \
  --sample-rate 0.01 \
  --parquet-batch-size 4096 \
  --seq-len 200 \
  --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 \
  --non-seq-bag-len 64 \
  --non-seq-array-reduction last \
  --epochs 3 \
  --lr 1e-3 \
  --amp \
  --num-workers 2 \
  --pin-memory \
  --debug-batches 5 \
  --eval-every-batches 1 \
  --train-metrics-every 1
```

注意：`--sample-rate` 是按文件采样，不是按样本采样。它适合快速烟测，不适合作正式评估。

### 5.3 一天数据正式训练观察

如果 debug 指标健康，可以跑一天数据正式观察：

```bash
python scripts/run_production_hdfs.py \
  --data-path hdfs://namenode:8020/path/to/day \
  --feature-file data/selectedfeaturefinal.txt \
  --parquet-batch-size 4096 \
  --seq-len 200 \
  --sequence-lens click_seq=100,impression_seq=200,buy_seq=200 \
  --non-seq-bag-len 64 \
  --non-seq-array-reduction last \
  --epochs 1 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --amp \
  --num-workers 2 \
  --pin-memory \
  --prefetch-factor 2 \
  --debug-batches 5 \
  --eval-every-batches 20 \
  --train-metrics-every 20 \
  --save-checkpoint
```

这版主要看：

- AUC 是否开始稳定高于 0.5。
- loss 是否下降。
- debug 指标是否正常。
- CPU 和 GPU 利用率是否改善。

## 6. 测试特征质量的观察重点

### 6.1 标签质量

必须先确认训练目标是否正确。

当前代码使用：

```text
label_click
```

需要确认：

- `label_click` 是否确实是要预测的二分类目标。
- 正样本率是否合理。
- 是否存在某些小时、某些分区全是 0 或全是 1。
- 是否存在标签延迟、曝光粒度与点击归因不一致等问题。

### 6.2 NS-token 特征质量

当前 NS 部分包括 `context_token` 和 `item_token`。

需要重点看：

- 稀疏 ID 字段是否大量为 0 或 -1。
- sparse bag 是否真的有有效元素。
- dense 字段是否有 NaN、Inf、极端大值。
- CTR/CVR、价格、销量、时间类字段是否需要不同 transform。

当前 dense 大多经过 `signed_log1p` 或保留原值。后续可以做字段级 transform，例如：

- count、price、sales：适合 log。
- ratio、CTR、CVR：可能不适合 log，适合 clip 或标准化。
- time diff：可能适合 log 或分桶。

### 6.3 Sequence 特征质量

当前三条行为序列是：

- `click_seq`
- `impression_seq`
- `buy_seq`

每条序列使用该分组中的第一个字段作为 backbone，并用 backbone 生成 mask。

需要重点看：

- 三条序列的 `seq_mask_mean` 是否合理。
- backbone 字段是否大量为空。
- 同一条序列内的附加字段长度是否和 backbone 大致一致。
- 截断策略 `tail` 是否符合“保留最近行为”的业务假设。
- buy/impression 序列长度 200 是否能覆盖足够历史。

如果某条序列 mask 长期接近 0，这条序列基本没有给模型提供有效信息。

## 7. 后续数据优化方向

### 7.1 做字段级统计报告

建议后续增加一个独立的数据 profiling 脚本，统计每个字段：

- 缺失率
- 默认值比例
- 非零率
- 唯一值数量
- top values
- dense 均值、方差、分位数
- 序列平均长度、P50、P90、P99

这一步收益很高，因为生产特征数量多，很多字段可能看起来有用，实际全是默认值。

### 7.2 检查 hash bucket 碰撞

当前 sparse 特征会按字段进入不同 bucket。后续需要观察：

- 高频 ID 是否碰撞严重。
- bucket size 是否过小。
- 不同字段是否需要不同 bucket size。

如果碰撞太重，模型会很难区分关键 item、cat、mall、user 行为。

### 7.3 时间切分验证集

当前 progressive eval 只能用于训练过程观察，不适合作最终效果判断。

建议正式实验使用时间切分：

- 例如前 23 小时训练，最后 1 小时验证。
- 或前一天训练，后一天验证。

CTR 场景里随机切分容易高估效果，时间切分更接近真实线上泛化。

### 7.4 类别不平衡处理

如果 `pos_rate` 很低，可以尝试：

```bash
--pos-weight <negative_count / positive_count>
```

但要注意：

- `pos_weight` 更直接影响 loss 和召回倾向。
- AUC 不一定会因为 `pos_weight` 立刻提升。
- 先确保标签和特征没问题，再调这个参数。

## 8. 后续模型优化方向

### 8.1 先做简单 baseline

建议实现一个 NS-only baseline：

- 只使用 context/item 两个 NS-token。
- 不使用 click/impression/buy 序列。
- 用简单 MLP 或轻量交互层做分类。

如果 NS-only baseline AUC 能明显高于 0.5，而 HyFormer 不行，说明序列接入或 HyFormer 结构需要排查。

如果 NS-only baseline 也接近 0.5，说明问题更可能在标签或特征质量。

### 8.2 做 Sequence ablation

建议依次比较：

1. NS only
2. NS + click
3. NS + impression
4. NS + buy
5. NS + click + impression + buy

这样可以判断每条行为序列是否真的贡献信号。

如果某条序列加入后效果下降，可能原因是：

- 序列 mask 不正确。
- 序列附加字段错位。
- 噪声行为太多。
- 序列长度过长导致噪声覆盖信号。

### 8.3 调整模型容量

当前模型参数主要集中在 embedding。

建议先做三档实验：

小模型：

```bash
--field-embed-dim 32 --d-model 96 --token-mlp-hidden 192 --ffn-hidden 128
```

当前模型：

```bash
--field-embed-dim 64 --d-model 128 --token-mlp-hidden 320 --ffn-hidden 128
```

稍大模型：

```bash
--field-embed-dim 64 --d-model 192 --token-mlp-hidden 384 --ffn-hidden 256
```

初期不建议盲目加很多 HyFormer 层。先确认特征有效、AUC 能动，再扩大模型。

### 8.4 NS-token 和 Sequence-token 使用 specific MLP

当前设计偏向共享 MLP，符合 HyFormer 统一异构特征的思路，也有利于控制参数量。

但生产数据中不同分组语义差异很大：

- context
- item
- click
- impression
- buy

后续可以考虑：

- 共享底层 MLP + group-specific adapter。
- NS-token 使用一套 MLP，Sequence-token 使用另一套 MLP。
- 每个语义组使用独立的小 MLP。

这类改动要在数据确认有效后再做，否则容易把数据问题误判成模型问题。

### 8.5 Query Generation 优化

HyFormer 的核心思想之一是用 NS 特征和序列摘要生成每条 sequence branch 的 query。

后续可以检查：

- query generation 是否真正使用了每条序列的 pooling summary。
- click/impression/buy 是否应该有不同 query 参数。
- 是否需要多个 query token 表示不同兴趣子空间。

可以尝试：

```bash
--num-queries-per-seq 2
```

但建议在基础 AUC 正常后再做。

## 9. 推荐实验顺序

建议按下面顺序推进：

1. 跑 `--debug-batches 5`，确认标签、mask、非零率、dense、logits 正常。
2. 做小样本过拟合测试，确认模型能记住小数据。
3. 跑一天数据，观察 AUC 是否稳定高于 0.5。
4. 做字段级 profiling，找缺失率高、默认值高、序列为空的字段。
5. 做 NS-only baseline。
6. 做 click/impression/buy 的 sequence ablation。
7. 再做模型容量、specific MLP、query 数量等结构优化。

当前最关键的判断标准是：修复数组解析后，debug 输出是否证明特征和标签真的有效进入模型。如果这个问题解决，AUC 应该至少开始偏离 0.5；如果仍然完全不动，就优先做小样本过拟合和 NS-only baseline 来定位问题。
