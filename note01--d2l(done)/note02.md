# 第二章 预备知识

- [张量的构造与访问](#张量的构造与访问)
    - [基础张量](#基础张量)
    - [行向量](#行向量)
    - [调整张量的尺寸](#调整张量的尺寸)
    - [读取张量的尺寸](#读取张量的尺寸)
    - [张量元素的索引](#张量元素的索引)
    - [张量的原地操作](#张量的原地操作)
    - [张量的拷贝](#张量的拷贝)
- [张量的运算](#张量的运算)
    - [张量的连结 (concatenate)](#张量的连结-concatenate)
    - [张量元素的加总](#张量元素的加总)
    - [张量元素的加总 (不降维)](#张量元素的加总-不降维)
    - [张量元素的逻辑判断](#张量元素的逻辑判断)
    - [张量元素的 $L_2$ 范数](#张量元素的-l_2-范数)
- [张量与其他数据类型的转换](#张量与其他数据类型的转换)
    - [由列表转为张量](#由列表转为张量)
    - [由张量转变为 `ndarray`](#由张量转变为-ndarray)
    - [由 `ndarray` 转变为张量](#由-ndarray-转变为张量)
    - [由张量元素转变为标量](#由张量元素转变为标量)


## 张量的构造与访问

### 基础张量

```python
tensor1 = torch.ones((2, 3))
tensor2 = torch.zeros((2, 3))
tensor3 = torch.randn((2, 3))
```

### 行向量

```python
x1 = torch.arange(6)        # x1 = tensor([0, 1, 2, 3, 4, 5])
x2 = torch.arange(6.0)      # x2 = tensor([0., 1., 2., 3., 4., 5.])
```

### 调整张量的尺寸

```python
x3 = x1.reshape((2, 3))       # x3 = tensor([[0, 1, 2], [3, 4, 5]])
```

### 读取张量的尺寸

```python
shape1 = x3.shape           # shape1 = torch.Size([2, 3])
shape2 = x3.numel()         # shape2 = 6
```

### 张量元素的索引

```python
x4 = torch.arange(3)
x4[-1]                      # tensor([2])
x4[1, 3]                    # tensor([1, 2])
x4[:]                       # tensor([0, 1, 2])
```

### 张量的原地操作

```python
x5 = torch.arange(3)
y5 = torch.arange(3)
x5[:] = x5 + y5             # 就地运算
x5 += y5                    # 就地运算
x5 = x5 + y5                # 非就地运算
```

### 张量的拷贝

```python
x6 = torch.arange(3)
y6 = A.clone()
```

## 张量的运算

### 张量的连结 (concatenate)

```python
x = torch.arange(3).reshape(1, 3)
y = torch.arange(3).reshape(1, 3)
torch.cat((x, y), dim=0)    # tensor([[0, 1, 2], [0, 1, 2]])
torch.cat((x, y), dim=1)    # tensor([[0, 1, 2, 0, 1, 2]])
```

### 张量元素的加总

```python
torch.arange(3).sum()       # tensor(3)
```

### 张量元素的加总 (不降维)

```python
torch.ones(2, 3).sum(axis=0)
                            # tensor([2., 2., 2.])    降维加总
torch.ones(2, 3).sum(axis=0, keepdim=True)
                            # tensor([[2., 2., 2.]])  不降维加总
```

### 张量元素的逻辑判断

```python
x7 = torch.ones(3)
x8 = torch.arange(3)
x7 == x8                    # tensor([False, True, False])
```

### 张量元素的 $L_2$ 范数

```python
torch.norm(torch.tensor([3., 4.]))
                            # tensor(5.)
```

## 张量与其他数据类型的转换

### 由列表转为张量

```python
vector4 = torch.tensor([1, 2, 3, 4, 5, 6])
tensor4 = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

### 由张量转变为 `ndarray`

```python
x9 = torch.arange(3)
x9.numpy()                  # array([0, 1, 2], dtype=int64)
```

* 张量与其生成的 `ndarray` 共享内存.

### 由 `ndarray` 转变为张量

```python
x10 = numpy.arange(3)
torch.tensor(x10)            # tensor([0, 1, 2], dtype=torch.int32)
```

### 由张量元素转变为标量

```python
a = torch.tensor([5.7])
a.item()                     # 5.7
```
