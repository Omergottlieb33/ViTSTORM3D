import torch
import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self, input_patch_size, embedding_dim, output_depth, output_height, output_width):
        super(CNNDecoder, self).__init__()

        self.input_patch_size = input_patch_size
        self.embedding_dim = embedding_dim
        self.output_depth = output_depth
        self.output_height = output_height
        self.output_width = output_width

        # Linear projection
        self.linear_proj = nn.Linear(embedding_dim, 1024) # increase feature capacity

        # Reshape for 3D convolutions
        self.reshape_size = (1, 8, 8, 8)  # Adjust based on patch size and desired initial feature map
        self.reshape_features = 128 # 1024 / (8 * 8) = 16. but we need 128 for upsample
        self.reshape_linear = nn.Linear(1024, self.reshape_features * 8 * 8 * 8)

        # Upsampling and convolutions
        self.upconv1 = nn.ConvTranspose3d(self.reshape_features, 64, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [B, P, E]
        x = self.linear_proj(x) # [B, P, 1024]
        x = torch.mean(x, dim=1) # [B, 1024]
        x = self.reshape_linear(x)
        x = x.view(x.size(0), self.reshape_features, 8, 8, 8) # [B, 128, 8, 8, 8]

        x = self.upconv1(x) # [B, 64, 16, 16, 16]
        x = torch.relu(self.conv1(x)) # [B, 64, 16, 16, 16]
        x = self.upconv2(x) # [B, 32, 32, 32, 32]
        x = torch.relu(self.conv2(x)) # [B, 32, 32, 32, 32]
        x = self.upconv3(x) # [B, 16, 64, 64, 64]
        x = torch.relu(self.conv3(x)) # [B, 16, 64, 64, 64]
        x = nn.functional.pad(x, (28, 29, 28, 29, 28, 29)) # [B, 16, 121, 121, 121]
        x = self.final_conv(x) # [B, 1, 121, 121, 121]
        x = torch.squeeze(x, dim=1) # [B, 121, 121, 121]
        return x

# Example usage
B = 2  # Batch size
P = 16 # Patch size
E = 256 # Embedding dimension
D = 121 # Output depth
H = 121 # Output height
W = 121 # Output width

decoder = CNNDecoder(input_patch_size=P, embedding_dim=E, output_depth=D, output_height=H, output_width=W)
input_tensor = torch.randn(B, P, E)
output_tensor = decoder(input_tensor)
print(output_tensor.shape) # Should be [B, 121, 121, 121]