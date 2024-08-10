import torch
import torch.nn as nn


def chose_encoder(encoder_type):
    if encoder_type == 'basic':
        return CNEncoder
    elif encoder_type == 'inception':
        return CNEncoderInceptionBlock
    elif encoder_type == 'squeeze':
        return CNEncoderSqueeze
    else:
        raise ValueError("Unsupported encoder type")


def activation_from_name(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'swish':
        return nn.SiLU()
    elif activation == 'none':
        return nn.Identity()
    else:
        raise ValueError("Unsupported activation function")


##########################################################################
# CNE: Color Naming Encoder v1.0
class CNEncoder(nn.Module):
    """
    Encoder layer for a Color Naming encoder. It is a simple convolutional layer with a ReLU activation and a max
    pooling layer. Its only purpose is to reduce the spatial dimensions of the input tensor
    (which will be the color naming maps) extracting meaningful features.
    The convolutional layer receives an input with the form (B, Cin, H, W) and returns an output with the form
    (B, Cout, H/pooling_factor, W/pooling_factor).
    """
    def __init__(self, in_channels, out_channels, pooling_factor=2, max_pooling=True, activation='relu'):
        super(CNEncoder, self).__init__()
        if pooling_factor % 2 != 0:
            raise ValueError("The pooling factor must be an even number.")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.activation = activation_from_name(activation)
        if max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=pooling_factor, stride=pooling_factor)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pooling_factor, stride=pooling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

##########################################################################
# CNE: Color Naming Encoder v2.0 Fire module type encoder
class CNEncoderSqueeze(nn.Module):
    """
    Encoder layer for a Color Naming encoder. For this version, we aim to mimic the Fire module architecture, used in
    the SqueezeNet architecture. The Fire module consists of a 1x1 convolutional layer to reduce the number of input
    channels, followed by two parallel branches: a 1x1 convolutional layer and a 3x3 convolutional layer. The output of
    both branches is concatenated along the channel dimension.
    In the SqueezeNet architecture, the ratio between the number of input channels and the number channels that
    come out of the squeeze layer is usually between 1:6 and 1:8, so we will use a similar ratio here.
    """
    def __init__(self, in_channels, out_channels, pooling_factor=2, max_pooling=True, activation='relu', squeeze_ratio=6):
        super(CNEncoderSqueeze, self).__init__()
        self.squeeze_ratio = squeeze_ratio
        if in_channels % self.squeeze_ratio != 0:
            raise ValueError("The number of input channels must be divisible by the squeeze ratio.")
        if pooling_factor % 2 != 0:
            raise ValueError("The pooling factor must be an even number.")
        if out_channels % 2 != 0:
            raise ValueError("The number of output channels must be divisible by 2.")

        self.activation = activation_from_name(activation)
        squeeze_planes = in_channels // self.squeeze_ratio

        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1, stride=1, padding=0)
        self.expand1x1 = nn.Conv2d(squeeze_planes, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.expand3x3 = nn.Conv2d(squeeze_planes, out_channels // 2, kernel_size=3, stride=1, padding=1)

        if max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=pooling_factor, stride=pooling_factor)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pooling_factor, stride=pooling_factor)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        x1 = self.activation(self.expand1x1(x))
        x2 = self.activation(self.expand3x3(x))
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)
        return x


##########################################################################
# CNE: Color Naming Encoder v3.0. Inception block type encoder
class CNEncoderInceptionBlock(nn.Module):
    """
    Encoder layer for a Color Naming encoder. Its purpose is the same as the CNEncoderLayer, but it uses an Inception
    block instead of a simple convolutional layer.
    """
    def __init__(self, in_channels, out_channels, pooling_factor=2, max_pooling=True, activation='relu'):
        super(CNEncoderInceptionBlock, self).__init__()
        if pooling_factor % 2 != 0:
            raise ValueError("The pooling factor must be an even number.")
        if out_channels % 4 != 0:
            raise ValueError("The number of output channels must be divisible by 4.")

        self.activation = activation_from_name(activation)

        branch_channels = out_channels // 4

        # 1x1 convolution branch
        self.conv1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0)

        # 3x3 convolution branch
        self.conv3x3_reduce = nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)

        # 5x5 convolution branch
        self.conv5x5_reduce = nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0)
        self.conv5x5 = nn.Conv2d(branch_channels, branch_channels, kernel_size=5, stride=1, padding=2)

        # Max pooling branch
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0)

        # Pooling layer to reduce the spatial dimensions
        if max_pooling:
            self.pooling = nn.MaxPool2d(kernel_size=pooling_factor, stride=pooling_factor)
        else:
            self.pooling = nn.AvgPool2d(kernel_size=pooling_factor, stride=pooling_factor)

    def forward(self, x):
        # 1x1 convolution branch
        x1 = self.activation(self.conv1x1(x))

        # 3x3 convolution branch
        x2 = self.activation(self.conv3x3_reduce(x))
        x2 = self.activation(self.conv3x3(x2))

        # 5x5 convolution branch
        x3 = self.activation(self.conv5x5_reduce(x))
        x3 = self.activation(self.conv5x5(x3))

        # Max pooling branch
        pool_proj = self.activation(self.pool_proj(self.pool(x)))
        x = torch.cat([x1, x2, x3, pool_proj], dim=1)

        x = self.pooling(x)
        return x

if __name__ == '__main__':
    # Test the CNEInceptionBlock
    x = torch.randn(2, 48, 128, 128).to('cuda')
    print("Input shape:", x.shape)  # Expected output: torch.Size([2, 48, 128, 128])
    print("Expected output shape:", torch.Size([2, 96, 64, 64]))
    encoder = CNEncoder(48, 96).to('cuda')
    out = encoder(x)
    print("Basic encoder output shape:", out.shape)  # Expected output: torch.Size([2, 96, 64, 64])
    encoder = CNEncoderSqueeze(48, 96).to('cuda')
    out = encoder(x)
    print("Squeeze encoder output shape:", out.shape)  # Expected output: torch.Size([2, 96, 64, 64])
    encoder = CNEncoderInceptionBlock(48, 96).to('cuda')
    out = encoder(x)
    print("Inception block encoder output shape:", out.shape)  # Expected output: torch.Size([2, 96, 64, 64])


