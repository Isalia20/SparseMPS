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

## Benchmarks

To reproduce the performance comparisons between dense and sparse operations on MPS, run the benchmark script:

```bash
python benchmark_dense_sparse.py