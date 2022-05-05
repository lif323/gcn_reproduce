import torch
def l2_matrix_norm(x):
    x = torch.reshape(x, (-1,))
    x = torch.sqrt(torch.sum(torch.pow(x, 2), axis=-1))
    return x 

if __name__ == "__main__":
    x = torch.tensor([[1, 2],[3, 4]], dtype=torch.float32)
    print(x)
    x = torch.reshape(x, shape=(-1,))
    print(torch.linalg.vector_norm(x))
    print(l2_matrix_norm(x))