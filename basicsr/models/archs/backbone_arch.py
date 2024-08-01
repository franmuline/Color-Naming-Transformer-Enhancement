
import torch
import torch.nn as nn

from basicsr.data.color_naming import ColorNaming
from basicsr.models.archs.backbone import Backbone
from basicsr.models.archs.prompt_ir_arch import PromptIR
from basicsr.models.archs.prompt_ir_color_naming_arch import PromptIRCN
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.restormer_color_naming_arch import RestormerCN


##########################################################################
##---------- Model with the Backbone -----------------------
class ModelCNBackbone(nn.Module):
    def __init__(self,
                 backbone,  ## Configuration for the Backbone
                 main_net,  ## Configuration for the Main Net
                 num_categories=6,  ## Number of categories for the color naming model
                 return_backbone=False,  ## Boolean to return the output of the Backbone
                 ):
        super(ModelCNBackbone, self).__init__()

        if backbone['type'] == 'Backbone':
            self.backbone = Backbone(**backbone['params'])
            if backbone['load_path']:
                self.backbone.load_state_dict(torch.load(backbone['load_path']))
            if backbone['freeze']:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("The backbone type is not supported.")

        if num_categories == 6 or num_categories == 11:
            self.color_naming = ColorNaming(num_categories=num_categories)
        elif num_categories == 0:
            self.color_naming = None
        else:
            raise ValueError("The number of categories is not supported.")

        if main_net['type'] == 'RestormerCN':
            assert self.color_naming is not None, "The color naming model must be initialized. Set num_categories to 6 or 11 or change the main_net type to Restormer."
            self.main_net = RestormerCN(**main_net['params'])
        elif main_net['type'] == 'PromptIRCN':
            assert self.color_naming is not None, "The color naming model must be initialized. Set num_categories to 6 or 11 or change the main_net type to PromptIR."
            self.main_net = PromptIRCN(**main_net['params'])
        elif main_net['type'] == 'Restormer':
            assert self.color_naming is None, "The color naming model must not be initialized. Set num_categories to 0 or change the main_net type to RestormerCN."
            self.main_net = Restormer(**main_net['params'])
        elif main_net['type'] == 'PromptIR':
            assert self.color_naming is None, "The color naming model must not be initialized. Set num_categories to 0 or change the main_net type to PromptIRCN."
            self.main_net = PromptIR(**main_net['params'])
        else:
            raise ValueError("The main_net type is not supported.")

        if main_net['load_path']:
            self.main_net.load_state_dict(torch.load(main_net['load_path']))
        if main_net['freeze']:
            for param in self.main_net.parameters():
                param.requires_grad = False

        self.return_backbone = return_backbone

    def forward(self, x):
        x_backbone = self.backbone(x)
        x_backbone_out = x_backbone.clone()
        if self.color_naming is not None:
            cn_probs = self.color_naming(x_backbone)
            # Color Naming returns a tensor with shape (C, B, H, W). We want (B, C, H, W)
            cn_probs = cn_probs.permute(1, 0, 2, 3)
            # We assert that the channels sum up to 1 (testing purposes)
            # assert torch.allclose(cn_probs.sum(dim=1), torch.ones_like(cn_probs.sum(dim=1))), "The sum of the color naming maps must be 1."
            cn_probs = cn_probs.float()
            # Concatenate the color naming maps to the input tensor
            x_backbone = torch.cat([x_backbone, cn_probs], dim=1)
        out = self.main_net(x_backbone)
        if self.return_backbone:
            return x_backbone_out, out
        return out
