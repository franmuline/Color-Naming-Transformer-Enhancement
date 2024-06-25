"""
color_naming.py - Contains the Joost van de Weijer et al. (2009) color naming model.

David Serrano (dserrano@cvc.uab.cat)
May 2024
"""

import os
import pathlib
from scipy.io import loadmat
import torch
from torch import tensor as to_tensor
from torchvision.transforms.functional import pil_to_tensor


class ColorNaming():
    def __init__(self, matrix_path=os.path.join(str(pathlib.Path(__file__).parent.resolve()), "joost_color_naming.mat"),
                 num_categories=6,
                 device='cuda'):
        """ Van de Weijer et al. (2009) Color Naming model python implementation.
        Van De Weijer, J. et al. Learning color names for real-world applications. IEEE Transactions on Image Processing
        The class is based on the MATLAB implementation by Van de Weijer et al. (2009) and it needs the w2c.mat original
        file. The input RGB image is converted to a set (6 or 11) color naming probability maps.

        If num_categories is 6: orange-brown-yellow, achromatic, pink-purple, red, green, blue.
        If num_categories is 11: black, blue, brown, gray, green, orange, pink, purple, red, white, yellow.
        """
        self.matrix = to_tensor(loadmat(matrix_path)['w2c']).to(device)
        self.num_categories = num_categories
        self.device = device

        if num_categories == 6:
            self.color_categories = [[2,5,10], [0,3,9], [6,7], [8], [4], [1]]
            self.color_categories = [torch.tensor(x).to(device) for x in self.color_categories]

    def __call__(self, input_tensor):
        """Converts an RGB image to a color naming image.

        Args:
        input_tensor: batch of RGB images (B x 3 x H x W)

        Returns:
            torch.tensor: Color naming image.
        """
        # Reconvert image to [0-255] range
        input_tensor = torch.clamp(input_tensor, 0, 1)
        img = (input_tensor * 255).int()

        index_tensor = torch.floor(
            img[:, 0, ...].view(img.shape[0], -1) / 8).long() + 32 * torch.floor(
            img[:, 1, ...].view(img.shape[0], -1) / 8).long() + 32 * 32 * torch.floor(
            img[:, 2, ...].view(img.shape[0], -1) / 8).long()

        prob_maps = []
        for w2cM in self.matrix.permute(*torch.arange(self.matrix.ndim-1, -1, -1)):
            out = w2cM[index_tensor].view(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3))
            prob_maps.append(out)
        prob_maps = torch.stack(prob_maps, dim=0)

        if self.num_categories == 11:
            return prob_maps

        elif self.num_categories == 6:
            category_probs = []  # prob maps for each color category. [0, 1]
            for category in self.color_categories:
                cat_tensors = torch.index_select(prob_maps, 0, category).sum(dim=0)
                category_probs.append(cat_tensors)

            category_probs = torch.stack(category_probs, dim=0)

            return category_probs


def create_custom_collate_fn(color_naming):
    def custom_collate_fn(batch):
        """Collate function to convert PIL images to color naming images.

        Args:
            batch: list of (image, label) tuples.

        Returns:
            torch.tensor: Color naming image.
        """
        images = [element['lq'] for element in batch]
        image_tensor = torch.stack(images, dim=0)
        cn_images = color_naming(image_tensor)
        # Change order of dimensions, it returns [C, B, H, W], we want [B, C, H, W]
        cn_images = cn_images.permute(1, 0, 2, 3)
        # Transform cn_images to FloatTensor
        cn_images = cn_images.float()
        # Concatenate the color naming image to the input tensor so that the final output is
        # [B, 3 + num_categories, H, W]
        new_batch = {'lq': torch.cat([image_tensor, cn_images], dim=1),
                     'gt': torch.stack([element['gt'] for element in batch], dim=0)}
        for key, value in batch[0].items():
            if key not in ['lq', 'gt']:
                new_batch[key] = [element[key] for element in batch]
        return new_batch

    return custom_collate_fn
