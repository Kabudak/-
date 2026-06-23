# 生产数据 HyFormer 适配说明

这份文档解释当前这一版为了让 HyFormer 在生产数据上跑起来所做的改动。

核心目标有两个：

1. 把旧版生产 parquet 数据转换成复现 HyFormer 模型可以直接吃的张量。
2. 在不大改 backbone 的前提下，让特征组织方式更贴近 HyFormer 论文里的思路：非序列特征生成 NS token，行为历史生成 sequence token，再由 query generation 和 query boosting 做交互。

注意：目前代码文件仍然保持无中文；这份 Markdown 是说明文档，所以使用中文。

## 一、当前整体设计

当前生产数据被分成五类：

| 原始分组 | 当前模型中的角色 | 说明 |
| --- | --- | --- |
| `context` | `context_token` | 非序列上下文 token |
| `item` | `item_token` | 非序列候选商品 token |
| `impr` | `impression_seq` | 曝光行为序列 |
| `click` | `click_seq` | 点击行为序列 |
| `buy` | `buy_seq` | 购买行为序列 |

这里最关键的改动是：`item` 不再作为一个序列分支，而是作为 NS token。也就是说，候选商品本身是当前样本的静态信息，不应该被模型当作用户历史行为的一段序列。

三个历史行为分支是：

```text
click_seq, impression_seq, buy_seq
```

两个非序列 token 是：

```text
context_token, item_token
```

模型的 Query Generation 当前使用：

```text
context_token + item_token + click_seq_pool + impression_seq_pool + buy_seq_pool
```

也就是用非序列信息和各条行为序列的池化摘要，一起生成每个序列分支的 query token。

## 二、这次改动过的文件

## 1. `data/selectedfeaturefinal.txt`

这个文件是手工整理后的生产特征分组文件。

当前分组如下：

| 分组 | 字段数量 | 用途 |
| --- | ---: | --- |
| `label` | 1 | 训练标签 |
| `context` | 20 | 上下文 NS token |
| `item` | 30 | 候选商品 NS token |
| `impr` | 20 | 曝光序列 |
| `click` | 17 | 点击序列 |
| `buy` | 20 | 购买序列 |

当前训练标签固定为：

```text
label_click
```

之前 `click` 里有 3 个和 `impr` 重复的字段，现在已经从 `click` 中删除，只保留在 `impr` 里：

```text
ups_impr_view_cat_index
ups_impr_view_id_index
ups_imprv2_clkv2_6h_cnt
```

这样做的原因是避免同一个原始字段同时进入两个序列分支，导致模型看到重复信息。

## 2. `utils/production_data.py`

这是当前生产数据预处理的核心文件。

它负责：

- 读取 parquet 文件。
- 读取 `selectedfeaturefinal.txt` 中的人工分组。
- 判断每个字段是稀疏 ID、连续数值、稀疏数组还是连续数值数组。
- 把非序列特征转换成 NS-token 所需张量。
- 把行为历史特征转换成 sequence-token 所需张量。
- 保存 metadata，供训练脚本自动还原模型输入结构。

当前固定标签列：

```text
LABEL_COLUMN = "label_click"
```

当前分组到模型结构的映射：

```text
context -> context_token
item    -> item_token
impr    -> impression_seq
click   -> click_seq
buy     -> buy_seq
```

如果特征文件里出现未知分组，代码会直接报错。这样做是为了防止分组名字写错以后，特征被静默丢掉或路由到错误位置。

## 3. `scripts/preprocess_production.py`

这是生产数据预处理入口。

当前推荐命令：

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0_final_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 100 --non-seq-bag-len 64
```

新增参数：

```text
--non-seq-bag-len
```

这个参数控制 NS-token 中稀疏数组特征最多保留多少个 ID。默认是 64。

例如某个 item 侧字段是一个商品 ID 数组，如果 `--non-seq-bag-len 64`，就最多保留 64 个 ID，超出的部分按截断策略处理。

## 4. `models/taac_hyformer.py`

模型包装层新增了 NS-token 稀疏数组处理能力。

之前模型只支持：

```text
non_seq_sparse
non_seq_dense
seq_sparse
seq_dense
seq_mask
```

现在模型还支持：

```text
non_seq_sparse_bag
non_seq_sparse_bag_mask
```

对于一个 NS-token 内部的稀疏数组字段，模型处理方式是：

1. 先把数组里的 ID bucket 化。
2. 用对应字段的 embedding table 查向量。
3. 用 mask 去掉无效位置和 0 ID。
4. 对有效 ID embedding 做 mean pooling。
5. 把 pooled vector 当作这个 NS-token 里面的一个稀疏特征。

这样做的好处是：数组特征不会把 `item` 膨胀成一条假序列，但又能保留数组里的多个 ID 信息。

## 5. `scripts/run_production.py`

生产训练脚本现在会加载以下张量：

```text
non_seq_sparse.pt
non_seq_sparse_bag.pt
non_seq_sparse_bag_mask.pt
non_seq_dense.pt
seq_sparse.pt
seq_dense.pt
seq_mask.pt
labels.pt
```

然后按新模型接口传入：

```text
model(
    non_seq_sparse,
    non_seq_sparse_bag,
    non_seq_sparse_bag_mask,
    non_seq_dense,
    seq_sparse,
    seq_dense,
    seq_mask,
)
```

为了兼容旧的预处理目录，如果旧目录里没有 `non_seq_sparse_bag.pt` 和 `non_seq_sparse_bag_mask.pt`，训练脚本会自动创建空 bag 张量，避免旧数据马上崩溃。

## 6. `scripts/run_baotao.py`

公开淘宝数据训练脚本也做了兼容。

因为旧版淘宝预处理没有 NS-token 数组 bag，所以这里给模型传入空的 bag 张量。

这样做是为了保持同一个 `TAACHyFormerClassifier` 接口，不需要维护两套模型 wrapper。

## 7. `Test.md`

命令示例已经更新为当前生产数据流程。

主要变化：

- 输入文件改为 `data/000000_0_final_head100.parquet`。
- 特征文件显式指定为 `data/selectedfeaturefinal.txt`。
- 新增 `--non-seq-bag-len 64`。
- 文档中说明当前标签来自 `label_click`。

## 三、当前预处理方式详解

## 1. 读取 parquet

`load_parquet_columns` 会用 `pyarrow` 读取 parquet。

读取后，每个 parquet 列会转换成 Python list：

```text
columns[field_name] = [row0_value, row1_value, row2_value, ...]
```

这么做的原因是生产数据里有很多嵌套数组字段，先转成 list 更容易统一处理。

## 2. 读取人工分组

`load_feature_schema` 会读取 `data/selectedfeaturefinal.txt`。

人工分组决定了一个字段进入哪里：

- `context` 和 `item` 进入非序列 token。
- `impr`、`click`、`buy` 进入行为序列。
- `label` 只用于生成训练标签。

当前非序列 token 不是按字段名自动猜出来的，而是优先尊重人工分组：

```text
context -> context_token
item    -> item_token
```

这一点很重要，因为手工分组比字段名规则更可信。

## 3. 非序列标量稀疏特征

标量稀疏特征一般是 ID 类字段，比如：

```text
site_id
scene_id
goods_id
cat1_id
mall_id
```

处理方式：

1. 取第一个标量值。
2. 转成整数。
3. 按字段类型选择 bucket size。
4. 计算 bucket ID。
5. 写入 `non_seq_sparse.pt`。

张量形状：

```text
[样本数, 非序列稀疏字段数]
```

轻量检查时得到：

```text
non_seq_sparse: (2, 15)
```

## 4. 非序列稀疏数组特征

这次重点处理的是 NS-token 中的数组特征。

有些 item 侧字段不是单个 ID，而是一组 ID，例如商品 ID 列表、类目 ID 列表、召回命中的 ID 列表等。它们不适合只取第一个值，也不适合作为单独行为序列。

所以当前把它们作为 sparse bag。

当前 sparse bag 字段有 12 个：

```text
flip_cat1_ids
flip_goods_ids
goods_name_bigram_hash
i2cat2_hit_ups_clk_tg
i2i_hit_clk_ids
i2i_hit_clk_ids_1d
i2i_hit_clk_ids_3d
i2i_hit_view_ids
i2i_list_swingv3gmv
list_clk_cat_ids_l20_x
list_clk_goods_ids
list_clk_mall_ids
```

预处理方式：

1. 把原始数组 flatten。
2. 按 `--non-seq-bag-len` 截断。
3. 默认使用 tail 截断，也就是保留末尾较新的值。
4. 把每个 ID bucket 化。
5. bucket 后非 0 的位置才算有效。
6. ID 写入 `non_seq_sparse_bag.pt`。
7. mask 写入 `non_seq_sparse_bag_mask.pt`。

张量形状：

```text
[样本数, sparse_bag字段数, non_seq_bag_len]
```

轻量检查使用 `--non-seq-bag-len 16`，得到：

```text
non_seq_bag:      (2, 12, 16)
non_seq_bag_mask: (2, 12, 16)
```

正常使用默认 `--non-seq-bag-len 64` 时，形状会类似：

```text
[样本数, 12, 64]
```

模型里对这些 sparse bag 做 mean pooling：

```text
ID embedding -> mask -> mean pooling -> 一个字段向量
```

这个字段向量再和其他 item 字段一起组成 `item_token`。

## 5. 非序列连续数值数组特征

连续数值数组不能用 ID embedding。

当前分成两类处理。

### 固定宽度数组

当前固定宽度数组：

```text
goods_cos_clk_sim_dis_cut3
goods_cos_view_sim_dis_cut3
u_clk_cnt_mix_d_kpos
```

处理方式是按位置展开。

例如宽度为 3 的字段会变成：

```text
field__pos0
field__pos1
field__pos2
```

这样能保留每个位置的含义。

### 统计摘要数组

当前统计摘要数组：

```text
i2cat2_hit_clk_timediff
i2i_hit_clk_timediff_3d
i2i_hit_clk_timediff_l10
ups_clk_hit_coclk_i2i_rank
ups_clk_hit_i2i_rank
ups_clk_hit_i2i_rank_1d
```

处理方式是对数组做统计摘要：

```text
length_log
mean
std
min
max
last
```

这些统计值会进入 `non_seq_dense.pt`。

轻量检查时得到：

```text
non_seq_dense: (2, 59)
```

这里的 59 比原始 `item` 的 30 个字段多，是因为数组字段被展开成了多个内部 dense 输入。原始人工分组没有变，只是模型输入维度变多了。

## 6. 序列特征

序列特征来自三个分支：

```text
click_seq
impression_seq
buy_seq
```

对每个分支里的字段，预处理会判断它是稀疏 ID 还是连续数值。

稀疏序列字段处理：

1. flatten 原始数组。
2. 按 `--seq-len` 截断。
3. bucket 化每个 ID。
4. 写入 `seq_sparse.pt`。
5. 更新 `seq_mask.pt`。

连续数值序列字段处理：

1. flatten 原始数组。
2. 按 `--seq-len` 截断。
3. 做 signed log1p 变换。
4. 写入 `seq_dense.pt`。
5. 更新 `seq_mask.pt`。

有些字段本身已经是 log 后的值，例如：

```text
log_all_impr_tg_1d
```

这类字段会跳过 signed log1p，避免重复 log。

轻量检查使用 `--seq-len 8`，得到：

```text
seq_sparse: (2, 3, 8, 41)
seq_dense:  (2, 3, 8, 16)
seq_mask:   (2, 3, 8)
```

正常使用 `--seq-len 100` 时，形状会类似：

```text
seq_sparse: [样本数, 3, 100, 41]
seq_dense:  [样本数, 3, 100, 16]
seq_mask:   [样本数, 3, 100]
```

其中第二维的 3 表示：

```text
click_seq, impression_seq, buy_seq
```

## 7. 标签

当前标签来自：

```text
label_click
```

处理后写入：

```text
labels.pt
```

当前是二分类任务，所以标签是 0 或 1。

## 8. metadata

预处理会保存 `metadata.json`。

里面记录：

- 样本数。
- 正负样本数。
- 正样本比例。
- 序列长度。
- sparse bag 长度。
- 序列分支顺序。
- 每个序列分支包含哪些字段。
- 非序列稀疏字段列表。
- 非序列 sparse bag 字段列表。
- 非序列 dense 展开字段列表。
- 稀疏字段 bucket cardinality。
- token group。
- 数组字段处理策略。

训练脚本会读取 metadata 来创建模型，所以 metadata 是当前流程里非常重要的一部分。

## 四、预处理效率优化

之前的预处理逻辑更接近：

```text
row -> branch -> field -> step
```

也就是先遍历每一行，再遍历每个分支，再遍历字段，再遍历序列位置。

这种写法在小数据上问题不大，但生产数据样本多、字段多、数组多，会比较慢。

现在改成更偏字段优先：

```text
field -> all rows
```

也就是说，每次拿一个字段，对所有样本生成一个完整的 NumPy 数组，然后一次性写入目标张量的对应切片。

当前主要 helper：

| 函数 | 用途 |
| --- | --- |
| `bucket_scalar_array` | 处理标量稀疏字段 |
| `extract_sparse_matrix` | 处理稀疏数组和稀疏序列 |
| `extract_dense_matrix` | 处理连续数值序列 |
| `extract_dense_spec_array` | 处理非序列 dense 标量、位置展开、统计摘要 |

这并不是完全向量化，因为 parquet 里的嵌套数组仍然需要逐行解析。但它已经去掉了最重的三层嵌套结构，对大数据会更友好。

## 五、当前输出文件

完整预处理后，输出目录里应该有：

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

训练入口 `scripts/run_production.py` 会读取这些文件。

## 六、已经做过的轻量验证

由于本机内存有限，没有跑训练。

已经通过的检查：

```bash
D:\torch\.venv\Scripts\python.exe -m py_compile main_pytorch.py models\taac_hyformer.py scripts\preprocess_production.py scripts\run_production.py scripts\run_baotao.py utils\production_data.py
```

2 行样本预处理检查：

```bash
D:\torch\.venv\Scripts\python.exe scripts\preprocess_production.py --input-parquet data\000000_0_final_head100.parquet --feature-file data\selectedfeaturefinal.txt --output-dir data\production_sample_check --max-rows 2 --seq-len 8 --non-seq-bag-len 16
```

输出形状：

```text
labels:           (2,) pos_rate=0.0000
non_seq_sparse:   (2, 15)
non_seq_bag:      (2, 12, 16)
non_seq_bag_mask: (2, 12, 16)
non_seq_dense:    (2, 59)
seq_sparse:       (2, 3, 8, 41)
seq_dense:        (2, 3, 8, 16)
seq_mask:         (2, 3, 8)
sequence_names:   ['click_seq', 'impression_seq', 'buy_seq']
token_groups:     ['context_token', 'item_token']
```

小模型前向检查：

```text
logits_shape (2, 1)
finite True
```

这说明当前预处理结果能被模型正常吃进去，并且输出没有 NaN 或 Inf。

## 七、到公司机器上的建议验证顺序

第一步，先跑完整预处理：

```bash
python scripts/preprocess_production.py --input-parquet data/000000_0_final_head100.parquet --feature-file data/selectedfeaturefinal.txt --output-dir data/production_sample --seq-len 100 --non-seq-bag-len 64
```

第二步，跑一个非常小的 smoke training：

```bash
python scripts/run_production.py --data-dir data/production_sample --epochs 1 --batch-size 16 --d-model 32 --ffn-hidden 64 --hyformer-layers 1 --short-seq-len 8
```

如果内存不够，优先调小这些参数：

1. `--batch-size`
2. `--seq-len`
3. `--non-seq-bag-len`
4. `--d-model`
5. `--field-embed-dim`

一般先降 batch size，最直接。

## 八、后续还需要确认的问题

后面到公司机器上主要确认这些点：

1. `label_click` 是否就是最终训练目标。
2. 完整生产数据能否顺利完成预处理。
3. 完整生产数据能否跑通 1 个 epoch。
4. sparse bag 当前用 mean pooling 是否足够，后续是否要改成 attention pooling。
5. `click_seq`、`impression_seq`、`buy_seq` 是否应该使用不同的序列长度。
6. 当前 bucket size 在完整数据上的冲突率是否可以接受。
7. 某些 item 侧数组是否应该拆成额外 NS token，而不是都放进 `item_token` 内部。

目前这一版的优先级是：先稳定跑通生产数据，再根据完整数据表现做更细的模型结构调整。
