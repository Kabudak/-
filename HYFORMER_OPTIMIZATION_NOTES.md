# HyFormer 生产训练改动说明与测试计划

本文档记录当前保留的代码改动、每项改动的目的、影响范围、回滚方式，以及下一步在公司生产机器上的测试方向。

## 当前保留的改动范围

本轮最终保留三类核心改动：

1. Arrow/NumPy 批处理预处理路径
2. 多 `data-path` 文件顺序与 shuffle 对照能力
3. MoE 可选 FFN 实现与单独 MoE 入口脚本

已按要求删除或撤回的内容：

- 没有保留 holdout / validation-mode / valid-data-path 相关评估口径改动
- 没有保留数据体检脚本
- 当前主训练脚本仍使用原来的 progressive per-batch 评估/训练流程

## 新增文件

### `utils/arrow_preprocess.py`

新增 Arrow/NumPy 批处理预处理工具，目标是减少 HDFS/Parquet 训练中的 Python 对象循环。

它主要负责：

- 从 Arrow `RecordBatch` 的 list 列中直接展开 values
- 使用 Arrow C++ kernel 获取 list parent indices
- 在 NumPy 中批量完成：
  - 标量抽取
  - sparse id bucket
  - dense signed log1p
  - 序列截断
  - 序列 padding
  - mask 构造
- 最后通过 `torch.from_numpy(...)` 转成 tensor

这替代了旧路径里的大量：

```python
record_batch.to_pandas()
batch_df[col].tolist()
for row in rows:
    ...
```

注意：这里不是从 Parquet 原始 buffer 到 Torch 的全链路 zero-copy。因为变长 list 必须被截断/填充成定长矩阵，这一步会 materialize 成新的 NumPy buffer。真正 zero-copy 的部分是 NumPy 到 Torch 的 `torch.from_numpy(...)`。

### `scripts/run_production_hdfs_moe.py`

新增 MoE 对比入口。

它不是复制一份完整训练逻辑，而是薄 wrapper：

- 默认设置 `--ffn-type moe`
- 默认输出目录为 `outputs/production_hdfs_moe`
- 其余训练流程复用 `scripts/run_production_hdfs.py`

这样 FFN 与 MoE 对比时，数据读取、预处理、progressive 评估、checkpoint、metadata 逻辑保持一致。

## 修改文件

### `production_hdfs_dataset.py`

#### 1. 新增 `preprocess_engine`

新增参数：

```text
preprocess_engine = "arrow" | "pandas"
```

命令行对应：

```bash
--preprocess-engine arrow
--preprocess-engine pandas
```

默认是 `arrow`。

含义：

- `arrow`：走新的 Arrow/NumPy 批处理路径
- `pandas`：走旧的 pandas/list/object 路径

如果 Arrow 路径遇到暂不支持的数据形态，默认会自动 fallback 到 pandas 路径，并打印一次 warning。

可以用下面参数关闭 fallback，使问题直接暴露：

```bash
--no-arrow-fallback
```

#### 2. 新增 schema 校验增强

原逻辑主要校验第一个 parquet 文件。现在会至少按 data source 取代表文件校验 required columns。

新增参数：

```bash
--validate-all-files
```

开启后会校验所有文件 schema。这个会增加启动开销，适合排查，不建议默认长期打开。

#### 3. 新增 file shuffle 控制

新增参数：

```text
file_shuffle = "global" | "none" | "source_round_robin"
```

命令行对应：

```bash
--file-shuffle global
--file-shuffle none
--file-shuffle source_round_robin
```

含义：

- `global`：旧行为。所有 data-path 的文件合并后全局文件级 shuffle
- `none`：按 `--data-path` 传入顺序读文件
- `source_round_robin`：每个 data source 内部 shuffle，然后 source 之间轮转取文件

这个改动只控制文件顺序，不做 row-level shuffle，也不做 batch-level shuffle。

#### 4. 新增文件顺序预览

新增：

```bash
--debug-file-order 20
```

用于打印当前 epoch 前 N 个即将读取的 parquet 文件。主要用于确认四小时多 path 输入时，训练实际看到的文件顺序是否符合预期。

### `scripts/run_production_hdfs.py`

保留原 progressive 训练流程。

新增参数：

```bash
--preprocess-engine
--no-arrow-fallback
--validate-all-files
--file-shuffle
--debug-file-order
--ffn-type
--moe-num-experts
--moe-top-k
--moe-shared-experts
--moe-expert-hidden
```

没有保留以下参数：

```bash
--valid-data-path
--validation-mode
--holdout-last-data-path
```

metadata 仍使用：

```json
{
  "training_mode": "progressive_per_batch",
  "progressive_results": [...]
}
```

checkpoint 命名仍使用：

```text
hyformer_hdfs_<timestamp>_test_auc<auc>.pt
```

### `main_pytorch.py`

新增 `MoEFeedForward`，并让 `QueryBoostMixer` 支持两种 FFN 类型：

```text
ffn_type = "swiglu" | "moe"
```

默认仍是原来的 FFN 路径：

```text
ffn_type = "swiglu"
```

MoE 当前实现：

- top-k routing
- 每个 expert 是 `SwiGLUFeedForward`
- 支持 shared experts
- MoE 替换的是 QueryBoostMixer 里的 channel FFN 部分

注意：当前 MoE 实现优先保证功能对齐和实验可比性，不是高度优化的 fused MoE kernel。它可能减少激活专家计算，但也会引入 Python 层 expert dispatch 开销。因此 MoE 的第一轮目标应该是对比效果和显存/吞吐趋势，而不是期待立刻获得极致速度。

### `models/taac_hyformer.py`

将 MoE 参数从任务 wrapper 传入 HyFormer backbone：

```python
ffn_type
moe_num_experts
moe_top_k
moe_shared_experts
moe_expert_hidden
```

默认不改变原模型行为。

### `Test.md`

更新了生产 HDFS streaming 命令：

- 增加 `--preprocess-engine arrow`
- 增加 file shuffle 对照说明
- 增加 MoE progressive 对比命令
- 删除 holdout 和 profiling 相关命令

## 已完成的本地验证

本地验证只用于证明代码路径能跑通，不用于判断 AUC。

已验证：

- `py_compile` 通过
- 原 FFN progressive smoke test 通过
- MoE progressive smoke test 通过
- MoE backbone forward 通过
- 完整 `TAACHyFormerClassifier` MoE forward 通过
- 本地 100 行真实 parquet 上，Arrow 输出和 pandas 输出逐 tensor 一致
- `--file-shuffle none` 和 `--debug-file-order` 能正常工作
- `git diff --check` 通过

其中 Arrow vs pandas 等价性验证结果：

```text
8 个 tensor 全部一致
最大差值均为 0.0
```

### 1. Arrow 预处理正确性灰度

目标：确认 Arrow 路径在生产 parquet 上不改变训练语义。

建议先在一个较小真实切片上跑两组：

```bash
--preprocess-engine pandas
```

和：

```bash
--preprocess-engine arrow --no-arrow-fallback
```

固定以下参数完全一致：

- data-path
- seed
- batch size
- model size
- learning rate
- eval-every-batches
- train-metrics-every
- num-workers

观察：

- debug batch 输出是否一致或接近
- label 分布是否正常
- `seq_mask_mean` 是否正常
- dense 是否 finite
- loss/AUC 曲线是否在同一量级
- Arrow 路径是否报错

判断：

- 如果 Arrow 和 pandas 曲线基本一致，说明预处理语义可信
- 如果 Arrow 报错，先保留 `--preprocess-engine pandas` 回退，再定位具体字段类型
- 如果 Arrow 能跑但指标明显不同，优先对比第一批 tensor 的关键统计

### 2. Arrow 预处理性能测试

目标：确认 CPU 预处理瓶颈是否缓解，GPU 利用率是否提升。

建议对比：

```bash
--preprocess-engine pandas
```

和：

```bash
--preprocess-engine arrow
```

观察：

- 每 batch wall time
- 每 100 batch 总耗时
- GPU utilization
- GPU memory
- CPU utilization
- DataLoader worker utilization
- active.tensor / tensor pipe active
- 是否出现 GPU 等数据的空窗

如果 Arrow 明显加速，再继续调：

```bash
--parquet-batch-size 4096
--parquet-batch-size 8192
--parquet-batch-size 16384
```

以及：

```bash
--num-workers 0
--num-workers 2
--num-workers 4
--num-workers 8
--pin-memory
--prefetch-factor 2
```

判断：

- CPU 仍满、GPU 仍低：继续增大 worker 或 batch size
- GPU 显存压力升高但吞吐提升：寻找 batch size 平衡点
- worker 多了反而慢：可能 HDFS I/O 或反序列化竞争，回退到较小 worker 数

### 3. 多 `data-path` 文件顺序测试

目标：判断四小时数据训练变差是否与文件级 shuffle / 文件顺序有关。

在同一份四小时数据上，只改变 `--file-shuffle`：

旧行为：

```bash
--file-shuffle global
```

按传参顺序读：

```bash
--file-shuffle none --debug-file-order 20
```

source 轮转：

```bash
--file-shuffle source_round_robin --debug-file-order 20
```

其他参数必须完全一致：

- seed
- batch size
- lr
- model size
- data-path
- eval-every-batches 5
- train-metrics-every
- num-workers
- preprocess-engine

观察：

- 每 5 batch eval AUC 曲线
- train loss 曲线
- train AUC 曲线
- debug-file-order 打印的文件顺序
- 前 100/500/1000 batch 的平均 AUC

判断：

- `global` 差，`none` 好：全局文件 shuffle 或流式顺序影响训练
- `none` 差，`source_round_robin` 好：需要跨小时/source 均衡喂数据
- 三者都差，而单小时好：更可能是四小时混合后的优化动态、学习率、batch 连续分布或模型容量问题

### 4. 多 worker 数据读取测试

目标：确认多 worker 不重复读、不漏读，并且真的提升吞吐。

当前 `IterableDataset` 已按 worker id 做 file sharding：

```text
worker 0: file_list[0::num_workers]
worker 1: file_list[1::num_workers]
...
```

建议对比：

```bash
--num-workers 0
--num-workers 2
--num-workers 4
```

同时开启：

```bash
--debug-file-order 20
```

观察：

- 总 batch 数是否符合预期
- 每个 epoch 耗时
- 是否出现 HDFS open retry 增多
- 是否出现某些 worker 明显拖慢

如果怀疑重复/漏读，需要额外在生产日志里记录每个 worker 的文件数和文件名摘要。当前代码已有 worker sharding，但没有逐 worker 文件摘要日志；如后续需要可以再补。

### 5. MoE 对比实验

目标：验证 MoE 替换 QueryBoostMixer FFN 后，对效果、吞吐、显存的影响。

FFN baseline：

```bash
python scripts/run_production_hdfs.py \
  --data-path ... \
  --ffn-type swiglu \
  ...
```

MoE：

```bash
python scripts/run_production_hdfs_moe.py \
  --data-path ... \
  --moe-num-experts 8 \
  --moe-top-k 2 \
  --moe-shared-experts 1 \
  ...
```

建议第一轮只改 `ffn_type`，其他参数保持一致。

然后再做 MoE 小矩阵：

```text
num_experts=4, top_k=1, shared=1
num_experts=4, top_k=2, shared=1
num_experts=8, top_k=1, shared=1
num_experts=8, top_k=2, shared=1
```

观察：

- progressive eval AUC
- train loss
- 每 batch 耗时
- GPU memory
- active.tensor
- 是否出现专家路由导致的吞吐下降

判断：

- AUC 提升但速度下降：说明 MoE 有效果潜力，后续考虑优化 dispatch
- AUC 不升且速度下降：先不继续 MoE，回到数据/优化问题
- AUC 接近但显存/吞吐更好：可以继续调 expert hidden/top-k

### 6. active.tensor 后续观察

active.tensor 低通常表示 Tensor Core 相关计算单元没有被充分使用。它可能来自：

- 数据预处理慢，GPU 等 CPU
- H2D copy 慢
- batch 太小
- AMP 没开或部分算子没有走 Tensor Core
- 模型里小矩阵/稀疏 embedding/控制流占比高
- 频繁 CPU metric 同步

当前改动已经优先处理：

- Arrow 预处理减少 CPU 对象循环
- `--pin-memory` + non-blocking transfer
- `--eval-every-batches` / `--train-metrics-every` 避免每 batch CPU 同步

下一步测试时应同时记录：

- GPU util
- active.tensor
- SM active
- H2D copy 时间
- DataLoader wait 时间
- batch wall time

如果 Arrow 后 GPU util 提升但 active.tensor 仍低，可能是模型算子结构本身 Tensor Core 占比不高，需要再看模型前向 profile。

### 7. 多 GPU 方向

当前没有实现 DDP。

建议顺序：

1. 先确认单 GPU 下 Arrow 路径能把 GPU utilization 提上去
2. 再确认四小时数据的读取/shuffle 问题
3. 再做 DDP

原因：

- 如果瓶颈仍在 CPU/HDFS，多 GPU 只会让多张卡一起等数据
- `IterableDataset` 做 DDP 时需要同时按 rank 和 worker 切分文件，否则容易重复读
- metric 聚合、checkpoint 只主 rank 保存、随机种子和文件顺序都需要重新设计

DDP 适合在单卡吞吐已经比较健康后再做。

## 推荐生产测试顺序

建议按下面顺序推进：

1. 单小时数据，`pandas` vs `arrow`
2. 单小时数据，Arrow + worker/pin/batch size 调优
3. 四小时数据，固定旧参数，仅比较 `file_shuffle`
4. 四小时数据，Arrow + 最优 `file_shuffle`
5. 同样数据下 FFN vs MoE
6. 如果单卡吞吐仍不足，再评估 DDP

## 快速回滚方式

如果线上需要最小风险回滚：

```bash
--preprocess-engine pandas
--file-shuffle global
--ffn-type swiglu
```

这会尽量接近原有行为，同时仍保留新的命令行能力。
