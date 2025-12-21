# SparseMPS

All the ops currently implemented in PyTorch MPS can be seen in `list_of_ops.txt` file. There are currently 162 ops implemented which is on par with other CPU/CUDA implementations in torch.

## Core Operations

### Matrix Operations
- **mm / bmm**: Matrix multiplication and batched matrix multiplication for sparse tensors
- **addmm**: Matrix-matrix multiplication with addition (computes β·input + α·(mat1 @ mat2))
- **mv**: Matrix-vector multiplication
- **_sparse_sparse_matmul**: Direct sparse-sparse matrix multiplication

### Sparse Tensor Construction & Management
- **_sparse_coo_tensor_with_dims**: Create sparse COO tensor with specified dimensions
- **_sparse_coo_tensor_with_dims_and_tensors**: Create sparse COO tensor from existing index/value tensors
- **sparse_resize_ / sparse_resize_and_clear_**: Resize sparse tensors in-place
- **_coalesce / is_coalesced / _coalesced_**: Coalesce duplicate indices in sparse tensors
- **_indices / _values / indices / values**: Access sparse tensor components
- **_to_sparse / _to_dense**: Convert between sparse and dense representations
- **sparse_mask / _sparse_mask_projection**: Apply sparsity patterns to tensors
- **copy_sparse_to_sparse_**: Copy data between sparse tensors

### Element-wise Math Operations
- **add / sub / mul / div**: Basic arithmetic operations
- **abs / neg / sign / signbit**: Sign and absolute value operations
- **exp / expm1 / log1p**: Exponential and logarithmic functions
- **sqrt / pow**: Power operations
- **sin / cos / tan / sinh / tanh**: Trigonometric and hyperbolic functions
- **asin / atan / asinh / atanh**: Inverse trigonometric functions
- **ceil / floor / round / trunc / frac**: Rounding operations
- **erf / erfinv**: Error function and its inverse

### Reduction Operations
- **sum / _sparse_sum_backward**: Summation with backward pass support
- **norm / native_norm**: Vector and matrix norms
- **any**: Logical reduction

### Neural Network Operations
- **relu**: Rectified linear unit activation
- **_sparse_softmax / _sparse_log_softmax**: Softmax operations optimized for sparse tensors
- **_sparse_softmax_backward_data / _sparse_log_softmax_backward_data**: Backward passes for softmax

### Utility Operations
- **clone / copy_ / zero_**: Tensor copying and initialization
- **cat**: Concatenate sparse tensors
- **index_select**: Select indices along a dimension
- **permute / unsqueeze / narrow_copy**: Tensor shape manipulation
- **nan_to_num / isnan / isinf / isposinf / isneginf**: NaN and infinity handling

## Pull Requests & Merge History

### Foundation & Core Infrastructure

| PR | Description |
|:---|:------------|
| [#157238](https://github.com/pytorch/pytorch/pull/157238) | **Enabling sparse matrices on PyTorch MPS backend** |
| [#161852](https://github.com/pytorch/pytorch/pull/161852) | Enable sparse testing for MPS in a proper way |
| [#163951](https://github.com/pytorch/pytorch/pull/163951) | Fixing of testing suite |

### Sparse Tensor Construction & Coalescing

| PR | Description |
|:---|:------------|
| [#159729](https://github.com/pytorch/pytorch/pull/159729) | Coalesce for sparse tensors |
| [#160223](https://github.com/pytorch/pytorch/pull/160223) | `indices` and `values` ops |
| [#160254](https://github.com/pytorch/pytorch/pull/160254) | Sparse coalesce support for more dtypes |

### Element-wise & Unary Operations

| PR | Description |
|:---|:------------|
| [#160839](https://github.com/pytorch/pytorch/pull/160839) | Sparse unary and `add` method |
| [#161846](https://github.com/pytorch/pytorch/pull/161846) | More unary functions |
| [#162349](https://github.com/pytorch/pytorch/pull/162349) | Sparse `mul` op |
| [#166711](https://github.com/pytorch/pytorch/pull/166711) | `erfinv` op |
| [#166801](https://github.com/pytorch/pytorch/pull/166801) | `exp` op |

### Matrix Operations

| PR | Description |
|:---|:------------|
| [#165232](https://github.com/pytorch/pytorch/pull/165232) | Sparse matmuls (`mm`, `bmm`, `addmm`, `mv`) |
| [#167013](https://github.com/pytorch/pytorch/pull/167013) | Sparse-sparse matmul |
| [#167908](https://github.com/pytorch/pytorch/pull/167908) | `mm` out sparse |

### Tensor Manipulation & Indexing

| PR | Description |
|:---|:------------|
| [#162007](https://github.com/pytorch/pytorch/pull/162007) | `torch.cat` op for sparse tensors |
| [#168154](https://github.com/pytorch/pytorch/pull/168154) | `permute` op for sparse tensors |
| [#169368](https://github.com/pytorch/pytorch/pull/169368) | `index_select` MPS sparse |

### Reduction & Norm Operations

| PR | Description |
|:---|:------------|
| [#162885](https://github.com/pytorch/pytorch/pull/162885) | Sparse MPS `any` |
| [#164961](https://github.com/pytorch/pytorch/pull/164961) | Sparse `norm` |
| [#169240](https://github.com/pytorch/pytorch/pull/169240) | Sparse MPS backward `sum` |

### Masking & Broadcasting

| PR | Description |
|:---|:------------|
| [#163694](https://github.com/pytorch/pytorch/pull/163694) | `unique_dim` and sparse broadcast ops |
| [#165102](https://github.com/pytorch/pytorch/pull/165102) | Sparse mask implementation |
| [#166260](https://github.com/pytorch/pytorch/pull/166260) | Sparse mask projection |
| [#168112](https://github.com/pytorch/pytorch/pull/168112) | Fixing of broadcasting issues on sparse MPS |

### Neural Network Operations

| PR | Description |
|:---|:------------|
| [#169125](https://github.com/pytorch/pytorch/pull/169125) | Sparse `softmax` / `log_softmax` |

---

**Total Merged PRs:** 24 | **Operations Implemented:** 162

## Basic Usage

```python
import torch

# Create a sparse COO tensor on MPS
indices = torch.tensor([[0, 1, 2],
                        [1, 0, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3), device="mps")

# Coalesce duplicate indices
indices_dup = torch.tensor([[0, 0, 1],
                            [1, 1, 0]])
values_dup = torch.tensor([1.0, 2.0, 3.0])
uncoalesced = torch.sparse_coo_tensor(indices_dup, values_dup, size=(2, 2), device="mps")
coalesced = uncoalesced.coalesce()  # Combines duplicates: [0,1] -> 1.0 + 2.0 = 3.0

# Sparse-dense matrix multiplication
sparse_matrix = torch.sparse_coo_tensor(
    torch.tensor([[0, 1, 2], [1, 2, 0]]),
    torch.tensor([1.0, 2.0, 3.0]),
    size=(3, 4),
    device="mps"
)
dense_matrix = torch.randn(4, 2, device="mps")
result = torch.mm(sparse_matrix, dense_matrix)

# Unary operations (applied only to non-zero values)
abs_sparse = torch.abs(sparse_tensor)
sin_sparse = torch.sin(sparse_tensor)
exp_sparse = torch.exp(sparse_tensor)
relu_sparse = torch.relu(sparse_tensor)
```

## Benchmarks

To reproduce the performance comparisons between dense and sparse operations on MPS, run the benchmark script:

```bash
python benchmark_dense_sparse.py