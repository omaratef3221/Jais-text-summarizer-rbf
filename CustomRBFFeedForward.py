import torch
import torch.nn as nn
from rbf_layer import RBFLayer  # Assuming RBFLayer is your custom RBF implementation

def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)

# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

# Multiquadric RBF
def multiquadric_rbf(r: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    return torch.sqrt(1 + (r / epsilon) ** 2)

# Inverse Multiquadric RBF
def inverse_multiquadric_rbf(r: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    return 1.0 / torch.sqrt(1 + (r / epsilon) ** 2)

# Linear RBF
def linear_rbf(r: torch.Tensor) -> torch.Tensor:
    return r

class CustomRBFFeedForward(nn.Module):
    def __init__(self, in_features, out_features, num_kernels):
        super(CustomRBFFeedForward, self).__init__()
        # RBFLayer from the given implementation
        self.rbf_layer = RBFLayer(
            in_features_dim=in_features,  # Input size (e.g., 5120)
            num_kernels=num_kernels,  # Number of kernels in the RBF layer (can be tuned)
            out_features_dim=out_features,  # Output size (e.g., 5120)
            radial_function=multiquadric_rbf,  # Use the Gaussian RBF
            norm_function=l_norm  # Use Euclidean norm
        )

    def forward(self, x):
        # Apply the RBF layer to the input x
        return self.rbf_layer(x)
