from CustomRBFFeedForward import *


def replace_ffn_with_rbf_jais(model, num_kernels):
    for i, block in enumerate(model.transformer.h):
        print(f"Replacing feedforward layers with RBF in block {i}")
        
        in_features = 1536
        intermediate_features = 4864  # You can adjust this based on the model's intermediate dimensions
        
        # Replace the first feedforward layer (c_fc) with an RBF layer
        block.mlp.c_fc = CustomRBFFeedForward(
            in_features=in_features,
            out_features=intermediate_features,
            num_kernels=num_kernels
        )
        
        # Replace the second feedforward layer (c_fc2) with an RBF layer, if used in this architecture
        block.mlp.c_fc2 = CustomRBFFeedForward(
            in_features=in_features,
            out_features=intermediate_features,
            num_kernels=num_kernels
        )
        
        # Replace the projection layer (c_proj) with an RBF layer
        block.mlp.c_proj = CustomRBFFeedForward(
            in_features=intermediate_features,
            out_features=in_features,
            num_kernels=num_kernels
        )
