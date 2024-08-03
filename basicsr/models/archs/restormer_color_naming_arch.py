"""
File added by Francisco Antonio Molina Bakhos.
Implements the modified version of Restormer's color naming model.
"""
import torch
import torch.nn as nn
from einops import rearrange
from .restormer_arch import LayerNorm, FeedForward, OverlapPatchEmbed, Downsample, Upsample, TransformerBlock, Restormer
from .arch_util import CustomSequential


def choose_sequential_type(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, boolean_cne, cn_only=False, cn_as_value=False):
    """
    Choose the type of Sequential layer to use in the model. If the boolean_cne is True, use the CustomSequential.
    The nn.Sequential module does not allow for multiple inputs and outputs in the forward method.
    Args:
        num_heads: number of heads in the multi-head attention mechanism
        ffn_expansion_factor: expansion factor of the feed-forward network
        bias: bias in the convolutional layers
        LayerNorm_type: type of LayerNorm to use
        num_blocks: number of Transformer blocks
        boolean_cne: boolean to include the color naming maps in the model
    Returns:
        nn.Sequential: Sequential layer with the Transformer blocks
    """
    if boolean_cne:
        return CustomSequential(
            *[TransformerBlockCN(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type, cn_only=cn_only, cn_as_value=cn_as_value) for i in range(num_blocks)])
    else:
        return nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])


def transformer_block_forward(inp_img_enc, inp_cn_enc, block, cne):
    """
    Forward method for the encoder part of the model. It receives the input image and the color naming maps and
    returns the output of the encoder.
    Args:
        inp_img_enc: Encoded image tensor
        inp_cn_enc: Encoded color naming tensor
        block: Transformer block module
        cne: Boolean to include the color naming maps in the model
    Returns:
        Output of the encoder
    """
    if cne:
        return block(inp_img_enc, inp_cn_enc)[0]
    else:
        return block(inp_img_enc)


##########################################################################
## CNE: Color Naming Encoder
class CNEncoderLayer(nn.Module):
    """
    Encoder layer for a Color Naming encoder. It is a simple convolutional layer with a ReLU activation and a max
    pooling layer. Its only purpose is to reduce the spatial dimensions of the input tensor
    (which will be the color naming maps) extracting meaningful features.
    The convolutional layer receives an input with the form (B, Cin, H, W) and returns an output with the form
    (B, Cout, H/pooling_factor, W/pooling_factor).
    """
    def __init__(self, in_channels, out_channels, pooling_factor=2, max_pooling=True, activation='relu'):
        super(CNEncoderLayer, self).__init__()
        if pooling_factor % 2 != 0:
            raise ValueError("The pooling factor must be an even number.")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError("Unsupported activation function")
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
## Color Naming Multi-DConv Head Transposed Self-Attention (MDTA)
class AttentionCN(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention (MDTA) modified to incorporate the encoded color naming maps
    in the attention mechanism.
    """
    def __init__(self, dim, num_heads, bias, cn_only=False, cn_as_value=False):
        super(AttentionCN, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Convolutional layers for the color naming maps
        self.cn_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.cn_conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.cn_only = cn_only
        self.cn_as_value = cn_as_value

    def forward(self, x, cn):
        assert x.shape == cn.shape, "The input tensor and the color naming tensor must have the same shape."
        b, c, h, w = x.shape

        cn = self.cn_conv3(self.cn_conv1(cn))

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        cn = rearrange(cn, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        if self.cn_as_value:
            if self.cn_only:
                v = cn
            else:
                v = v + cn
        else:
            cn = torch.nn.functional.normalize(cn, dim=-1)
            if self.cn_only:
                q = cn
            else:
                q = q + cn
                q = torch.nn.functional.normalize(q, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Modified Transformer Block to include the Color Naming maps
class TransformerBlockCN(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, cn_only=False, cn_as_value=False):
        super(TransformerBlockCN, self).__init__()

        # Normalization layer for the image features
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # Normalization layer for the color naming features
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionCN(dim, num_heads, bias, cn_only=cn_only, cn_as_value=cn_as_value)
        # Normalization layer for the output of the attention mechanism
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, cn):
        x = x + self.attn(self.norm1(x), self.norm2(cn))
        x = x + self.ffn(self.norm3(x))

        return x, cn


##########################################################################
##---------- Restormer with Color Naming maps -----------------------
class RestormerCN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim = 48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 boolean_cne=[True, True, True, True, False, False],  ## Boolean list to include the CNE layers in the encoder part of the model
                 cn_only=False,  ## Boolean to use CN maps ONLY as query or value in attention (it is summed to query or value otherwise)
                 cn_as_value=False,  ## Boolean to use the color naming maps as the value in the attention mechanism. It is used as the query otherwise
                 max_pooling=True,  ## Boolean to include max pooling in the CNE layers. Avg pooling is used otherwise
                 cne_activation='relu',
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(RestormerCN, self).__init__()

        if len(boolean_cne) < 6:
            # Add false values to the boolean_cne list until it reaches 7 elements
            boolean_cne += [False] * (6 - len(boolean_cne))
        elif len(boolean_cne) > 6:
            raise ValueError("The boolean_cne list must have 6 elements or less. It is assumed that if the list has less"
                             " than 6 elements, the rest of the elements are False.")

        # Assert that there is at least one CNE layer in the model
        assert any(boolean_cne), ("There must be at least one CNE layer in the model. Use the original Restormer model "
                                  "if color mappings are not going to be used.")

        self.boolean_cne = boolean_cne

        self.image_patch_embed = OverlapPatchEmbed(3, dim)
        self.cn_patch_embed = OverlapPatchEmbed(inp_channels - 3, dim)

        self.encoder_level1 = choose_sequential_type(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[0], boolean_cne[0], cn_only=cn_only, cn_as_value=cn_as_value)

        self.cne_1_2 = CNEncoderLayer(int(dim), int(dim*2**1), max_pooling=max_pooling, activation=cne_activation)  # First CNE layer (from level 1 to level 2)

        self.down_1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = choose_sequential_type(int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1], boolean_cne[1], cn_only=cn_only, cn_as_value=cn_as_value)

        self.cne_2_3 = CNEncoderLayer(int(dim * 2 ** 1), int(dim * 2 ** 2), max_pooling=max_pooling, activation=cne_activation)  # Second CNE layer (from level 2 to level 3)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = choose_sequential_type(int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2], boolean_cne[2], cn_only=cn_only, cn_as_value=cn_as_value)

        self.cne_3_4 = CNEncoderLayer(int(dim * 2 ** 2), int(dim * 2 ** 3), max_pooling=max_pooling, activation=cne_activation)  # Third CNE layer (from level 3 to level 4)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = choose_sequential_type(int(dim*2**3), heads[3], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[3], boolean_cne[3], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = choose_sequential_type(int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2], boolean_cne[4], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = choose_sequential_type(int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1], boolean_cne[5], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        # Assert that input batch has to have more than 3 channels
        # (3 for the image and the rest for the color naming maps)
        assert inp.shape[1] > 3, ("The input tensor must have more than 3 channels. Use the original Restormer "
                                  "model for 3-channel inputs.")
        inp_img, cn_maps = inp[:, :3, ...], inp[:, 3:, ...]

        inp_img_enc_level1 = self.image_patch_embed(inp_img)
        inp_cn_enc_level1 = self.cn_patch_embed(cn_maps)

        out_enc_level1 = transformer_block_forward(inp_img_enc_level1, inp_cn_enc_level1, self.encoder_level1, self.boolean_cne[0])

        inp_img_enc_level2 = self.down_1_2(out_enc_level1)
        inp_cn_enc_level2 = self.cne_1_2(inp_cn_enc_level1)

        out_enc_level2 = transformer_block_forward(inp_img_enc_level2, inp_cn_enc_level2, self.encoder_level2, self.boolean_cne[1])

        inp_img_enc_level3 = self.down2_3(out_enc_level2)
        inp_cn_enc_level3 = self.cne_2_3(inp_cn_enc_level2)

        out_enc_level3 = transformer_block_forward(inp_img_enc_level3, inp_cn_enc_level3, self.encoder_level3, self.boolean_cne[2])

        inp_img_enc_level4 = self.down3_4(out_enc_level3)
        inp_cn_enc_level4 = self.cne_3_4(inp_cn_enc_level3)

        latent = transformer_block_forward(inp_img_enc_level4, inp_cn_enc_level4, self.latent, self.boolean_cne[3])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = transformer_block_forward(inp_dec_level3, inp_cn_enc_level3, self.decoder_level3, self.boolean_cne[4])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = transformer_block_forward(inp_dec_level2, inp_cn_enc_level2, self.decoder_level2, self.boolean_cne[5])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img[:, :3, ...]

        return out_dec_level1


if __name__ == "__main__":
    model = RestormerCN(inp_channels=6, cne_activation='prelu').to('cuda')
    print(model)

    # Test the model
    # Generate random input tensor with 6 channels with shape (1, 6, 256, 256) and values between 0 and 1
    inp = torch.rand((1, 6, 128, 128)).to('cuda')
    out = model(inp)
