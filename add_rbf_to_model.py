from CustomRBFFeedForward import *

def replace_ffn_with_rbf_jais(model, num_kernels):
    for i, block in enumerate(model.transformer.h):
        print(f"Replacing feedforward layers with RBF in block {i}")
        
        in_features = 2048  # Adjusted to match the model architecture
        intermediate_features = 6144  # Typically 3x the in_features size
        

        # Get the device of the original layers
        original_device = block.mlp.c_fc.weight.device

        # Replace the first feedforward layer (c_fc) with an RBF layer and move it to the same device
        block.mlp.c_fc = CustomRBFFeedForward(
            in_features=in_features,
            out_features=intermediate_features,
            num_kernels=num_kernels
        ).to(original_device)

        # Replace the second feedforward layer (c_fc2) with an RBF layer and move it to the same device
        block.mlp.c_fc2 = CustomRBFFeedForward(
            in_features=in_features,
            out_features=intermediate_features,
            num_kernels=num_kernels
        ).to(original_device)

        # Replace the projection layer (c_proj) with an RBF layer and move it to the same device
        block.mlp.c_proj = CustomRBFFeedForward(
            in_features=intermediate_features,
            out_features=in_features,
            num_kernels=num_kernels
        ).to(original_device)

        print(f"RBF layers in block {i} moved to device: {original_device}")

