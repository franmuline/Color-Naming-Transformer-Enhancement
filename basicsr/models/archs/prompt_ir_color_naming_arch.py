"""
File added by Francisco Antonio Molina Bakhos.
Implements the modified version of PromptIR's color naming model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.restormer_arch import TransformerBlock, OverlapPatchEmbed, Downsample, Upsample
from basicsr.models.archs.restormer_color_naming_arch import CNEncoderLayer, choose_sequential_type, transformer_block_forward
from basicsr.models.archs.prompt_ir_arch import PromptGenBlock


##########################################################################
##---------- PromptIR with Color Naming -----------------------

class PromptIRCN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 boolean_cne_prompt=[True, True, True],  # Ordered by depth, from shallow to deep, not by their order in the network.
                 boolean_cne_encoder=[True, True, True, True, False, False],
                 cn_only=False,
                 cn_as_value=False,
                 max_pooling=False,
                 cne_activation='relu',
                 use_cne_in_pgm=True,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 decoder=False,
                 ):

        super(PromptIRCN, self).__init__()

        if len(boolean_cne_encoder) < 6:
            # Add false values to the boolean_cne list until it reaches 7 elements
            boolean_cne_encoder += [False] * (6 - len(boolean_cne_encoder))
        elif len(boolean_cne_encoder) > 6:
            raise ValueError(
                "The boolean_cne list must have 6 elements or less. It is assumed that if the list has less"
                " than 6 elements, the rest of the elements are False.")

        assert len(boolean_cne_prompt) == 3, "There should be a list of 3 boolean values for the color naming encoder"
        assert len(boolean_cne_encoder) == 6, "There should be a list of 6 boolean values for the color naming encoder"
        assert any(boolean_cne_prompt) or any(boolean_cne_encoder), "There must be at least one True value in the " \
                                                                        "boolean_cne_prompt or boolean_cne_encoder lists. Otherwise, use the PromptIR, the Restormer or the RestormerCN models."

        self.boolean_cne_prompt = boolean_cne_prompt
        self.boolean_cne_encoder = boolean_cne_encoder
        self.use_cne_in_pgm = use_cne_in_pgm

        assert (inp_channels == 9 or inp_channels == 14), "The input channels should be 9 or 14 for the color naming encoder"

        self.image_patch_embed = OverlapPatchEmbed(3, dim)
        self.cn_patch_embed = OverlapPatchEmbed(inp_channels - 3, dim)

        self.decoder = decoder

        if boolean_cne_prompt[0] or boolean_cne_prompt[1] or boolean_cne_prompt[2] \
                or boolean_cne_encoder[1] or boolean_cne_encoder[2] or boolean_cne_encoder[3] or \
                boolean_cne_encoder[4] or boolean_cne_encoder[5]:
            self.cne_1_2 = CNEncoderLayer(int(dim), int(dim * 2 ** 1), max_pooling=max_pooling,
                                          activation=cne_activation)  # First CNE layer (from level 1 to level 2)
        if boolean_cne_prompt[1] or boolean_cne_prompt[2] or boolean_cne_encoder[2] or boolean_cne_encoder[3] or \
                boolean_cne_encoder[4]:
            self.cne_2_3 = CNEncoderLayer(int(dim * 2 ** 1), int(dim * 2 ** 2), max_pooling=max_pooling,
                                          activation=cne_activation)
        if boolean_cne_prompt[2] or boolean_cne_encoder[3]:
            self.cne_3_4 = CNEncoderLayer(int(dim * 2 ** 2), int(dim * 2 ** 3), max_pooling=max_pooling,
                                          activation=cne_activation)

        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = choose_sequential_type(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[0], boolean_cne_encoder[0], cn_only=cn_only, cn_as_value=cn_as_value)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.encoder_level2 = choose_sequential_type(int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1], boolean_cne_encoder[1], cn_only=cn_only, cn_as_value=cn_as_value)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.encoder_level3 = choose_sequential_type(int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2], boolean_cne_encoder[2], cn_only=cn_only, cn_as_value=cn_as_value)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = choose_sequential_type(int(dim*2**3), heads[3], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[3], boolean_cne_encoder[3], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 512, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 512, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = choose_sequential_type(int(dim*2**2), heads[2], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[2], boolean_cne_encoder[4], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) + 224, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) + 224, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = choose_sequential_type(int(dim*2**1), heads[1], ffn_expansion_factor, bias, LayerNorm_type, num_blocks[1], boolean_cne_encoder[5], cn_only=cn_only, cn_as_value=cn_as_value)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1) + 64, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 64, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp, noise_emb=None):
        # Assert that input batch has to have more than 3 channels
        # (3 for the image and the rest for the color naming maps)
        assert inp.shape[1] > 3, ("The input tensor must have more than 3 channels. Use the original Restormer "
                                  "model for 3-channel inputs.")
        inp_img, cn_maps = inp[:, :3, ...], inp[:, 3:, ...]

        inp_img_enc_level1 = self.image_patch_embed(inp_img)
        inp_cn_enc_level1 = self.cn_patch_embed(cn_maps)

        out_enc_level1 = transformer_block_forward(inp_img_enc_level1, inp_cn_enc_level1, self.encoder_level1, self.boolean_cne_encoder[0])
        if hasattr(self, 'cne_1_2'):
            inp_cn_enc_level2 = self.cne_1_2(inp_cn_enc_level1)
        else:
            inp_cn_enc_level2 = None

        inp_img_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = transformer_block_forward(inp_img_enc_level2, inp_cn_enc_level2, self.encoder_level2, self.boolean_cne_encoder[1])
        if hasattr(self, 'cne_2_3'):
            inp_cn_enc_level3 = self.cne_2_3(inp_cn_enc_level2)
        else:
            inp_cn_enc_level3 = None

        inp_img_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = transformer_block_forward(inp_img_enc_level3, inp_cn_enc_level3, self.encoder_level3, self.boolean_cne_encoder[2])
        if hasattr(self, 'cne_3_4'):
            inp_cn_enc_level4 = self.cne_3_4(inp_cn_enc_level3)
        else:
            inp_cn_enc_level4 = None

        inp_img_enc_level4 = self.down3_4(out_enc_level3)
        latent = transformer_block_forward(inp_img_enc_level4, inp_cn_enc_level4, self.latent, self.boolean_cne_encoder[3])
        if self.decoder:
            if self.boolean_cne_prompt[2] and self.use_cne_in_pgm:
                dec3_param = self.prompt3(inp_cn_enc_level4)
                latent = torch.cat([latent, dec3_param], 1)
            elif self.boolean_cne_prompt[2] and not self.use_cne_in_pgm:
                dec3_param = self.prompt3(latent)
                latent = torch.cat([inp_cn_enc_level4, dec3_param], 1)
            else:
                dec3_param = self.prompt3(latent)
                latent = torch.cat([latent, dec3_param], 1)

            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = transformer_block_forward(inp_dec_level3, inp_cn_enc_level3, self.decoder_level3, self.boolean_cne_encoder[4])
        if self.decoder:
            if self.boolean_cne_prompt[1] and self.use_cne_in_pgm:
                dec2_param = self.prompt2(inp_cn_enc_level3)
                out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            elif self.boolean_cne_prompt[1] and not self.use_cne_in_pgm:
                dec2_param = self.prompt2(out_dec_level3)
                out_dec_level3 = torch.cat([inp_cn_enc_level3, dec2_param], 1)
            else:
                dec2_param = self.prompt2(out_dec_level3)
                out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)

            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = transformer_block_forward(inp_dec_level2, inp_cn_enc_level2, self.decoder_level2, self.boolean_cne_encoder[5])
        if self.decoder:
            if self.boolean_cne_prompt[0] and self.use_cne_in_pgm:
                dec1_param = self.prompt1(inp_cn_enc_level2)
                out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            elif self.boolean_cne_prompt[0] and not self.use_cne_in_pgm:
                dec1_param = self.prompt1(out_dec_level2)
                out_dec_level2 = torch.cat([inp_cn_enc_level2, dec1_param], 1)
            else:
                dec1_param = self.prompt1(out_dec_level2)
                out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)

            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
