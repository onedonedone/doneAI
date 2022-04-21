# 第三章 线性神经网络

- [从零搭建线性神经网络](#从零搭建线性神经网络)
    - [数据集生成](#数据集生成)
        - [`torch.normal`: 正态分布](#torchnormal-正态分布)
        - [`tensor.reshape((-1, 1))`: 向量转矩阵](#tensorreshape-1-1-向量转矩阵)
    - [重要的算法](#重要的算法)
- [线性回归的简洁实现](#线性回归的简洁实现)
    - [数据预处理](#数据预处理)
        - [`torch.utils.data.TensorDataset`: 数据打包](#torchutilsdatatensordataset-数据打包)
        - [`torch.utils.data.DataLoader`: 生成批量迭代器](#torchutilsdatadataloader-生成批量迭代器)
    - [初始化模型](#初始化模型)
        - [搭建单层全连接层模型](#搭建单层全连接层模型)
        - [模型参数的初始化](#模型参数的初始化)
    - [初始化训练参数](#初始化训练参数)
        - [定义损失函数](#定义损失函数)
        - [定义优化函数](#定义优化函数)
    - [训练过程](#训练过程)

## 从零搭建线性神经网络

### 数据集生成

使用线性模型参数: 权重 $\mathbf{w}=[w_1, w_2]$, 偏置 $b$, 噪声 $\epsilon$ 生成数据集

$$\mathbf{y}=\mathbf{Xw}+b+\epsilon$$

* 代码中用 `labels` 代表 $\mathbf{y}$, 用 `features` 代表 $\mathbf{X}$.

#### `torch.normal`: 正态分布

```python
tensor1 = torch.normal(1, 1, (3, 3))
tensor2 = torch.normal(mean=1, std=1, size=(3, 3))
```

#### `tensor.reshape((-1, 1))`: 向量转矩阵

```python
weights = torch.tensor([1, 2, 3])
weights.size()              # torch.Size([3])
weights = torch.T
weights.size()              # torch.Size([3])
weights = weights.reshape((-1, 1))
weights.size()              # torch.Size([3, 1])
weights = weights.T
weights.size()              # torch.Size([1, 3])
```

### 重要的算法

1. 数据批量读取

2. 更新参数

    1. 进入 `no_grad` 环境
    2. 使用梯度更新参数
    3. 梯度置零

3. 训练算法

    1. 后向传播
    2. 前向传播
    3. 更新参数

## 线性回归的简洁实现

### 数据预处理

#### `torch.utils.data.TensorDataset`: 数据打包

```python
features = torch.tensor([[1, 2], [2, 3], [3, 4]])
labels = torch.tensor([[1], [2], [3]])
dataset = torch.utils.data.TensorDataset(features, labels)

next(iter(dataset))         # (tensor([1, 2]), tensor([1]))
```

* 在 `dataset` 迭代器中, 每次迭代寻访 1 组数据.

#### `torch.utils.data.DataLoader`: 生成批量迭代器

```python
dataIteration = data.DataLoader(dataset, batchSize, shuffle=True)
```

* 在 `dataIteration` 迭代器中, 每次迭代寻访 `batchSize` 组数据.

### 初始化模型

#### 搭建单层全连接层模型

```python
net = torch.nn.Sequential(torch.nn.Linear(inputSize, outputSize))
```

#### 模型参数的初始化

```python
net[0].weight.data.normal_(mean=0, std=0.01)
net[0].bias.data.fill_(0)
```

### 初始化训练参数

#### 定义损失函数

```python
loss = torch.nn.MSELoss()   # 均方误差
```

#### 定义优化函数

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

### 训练过程

```python
epochs = 30
for epoch in range(epochs):
    for X, y in dataIteration:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
```