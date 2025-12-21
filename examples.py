import torch

def main():
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

if __name__ == '__main__':
    main()
