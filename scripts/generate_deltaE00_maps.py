"""
Visualize the deltaE00 maps between two images, usually the ground truth and the predicted image. It shows the deltaE00
values for each pixel in the image.
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from basicsr.metrics.deltaE_lpips import deltaE00

def find_image_path(directory, image_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if image_name in file and 'gt' not in file:
                return os.path.join(root, file)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_names', type=str, default=['Restormer_FiveK_baseline_official_split_2', 'RCNBackbone_pre_unf_FiveK_official_split_1', 'PromptIR_FiveK_baseline_official_split_1','PIRCNBackbone_pre_unf_FiveK_official_split_2'])
    parser.add_argument('--model_names', type=str, default=['AdaInt', 'NamedCurves', 'RCNBackbone_pre_unf_FiveK_official_split_1','PIRCNBackbone_pre_unf_FiveK_official_split_2'])  # ['Restormer_FiveK_baseline_official_split_2', 'RCNBackbone_pre_unf_FiveK_official_split_1', 'PromptIR_FiveK_baseline_official_split_1','PIRCNBackbone_pre_unf_FiveK_official_split_2'])
    parser.add_argument('--image_name', type=str, default='a4768')
    parser.add_argument('--output_dir', type=str, default='deltaE00_maps')
    args = parser.parse_args()
    value = 30
    # Create output directory
    # output_dir = os.path.join(args.output_dir, args.model_name)
    # os.makedirs(output_dir, exist_ok=True)

    # Load images
    input_path = f'../../Datasets/FiveK/input/'
    gt_path = f'../../Datasets/FiveK/expertC_gt/'
    pred_path_list = [f'../results/{model_name}/visualization/TestSet/' for model_name in args.model_names]

    # Find the image path for each directory. The image name may vary between directories, but it always contains
    # the name provided in the arguments.
    input_img_path = find_image_path(input_path, args.image_name)
    gt_img_path = find_image_path(gt_path, args.image_name)
    pred_img_path_list = [find_image_path(pred_path, args.image_name) for pred_path in pred_path_list]

    # Load images
    input_img = Image.open(input_img_path)
    gt_img = Image.open(gt_img_path)
    # Always read in RGB mode, as the deltaE00 function expects the images to be in RGB mode. Some images may be in
    # RGBA mode, and this will raise an error.
    pred_img_list = []
    for i, pred_img_path in enumerate(pred_img_path_list):
        pred_img = Image.open(pred_img_path)
        if pred_img.mode == 'RGBA':
            pred_img = pred_img.convert('RGB')
        pred_img_list.append(pred_img)

    # Calculate deltaE00 maps between the predicted images and the ground truth
    dE00 = deltaE00(return_map=True)
    deltaE00_maps = []
    deltaE00_values = []
    for pred_img in pred_img_list:
        deltaE00_map, deltaE00_value = dE00(pred_img, gt_img)
        deltaE00_maps.append(deltaE00_map)
        deltaE00_values.append(deltaE00_value)

    # For visualization, we are goingto plot everything in the following manner:
    # In the first column, we are going to plot the input image and the ground truth image, first and second row respectively.
    # In the following columns, we are going to plot the predicted image and the deltaE00 map for each model, first and second row respectively.
    # At the end of the plot, we are going to put a colorbar to show the deltaE00 values. This colorbar is going to be shared between all the deltaE00 maps.
    # All images are together, without any separation between them, no white space between images.
    # Only the columns where the deltaE00 maps are going to be plotted are going to have a title, the name of the model.
    print(f'DeltaE00 values: {deltaE00_values}')
    aspect_ratio = input_img.size[1] / input_img.size[0]
    rows = 2
    cols = 1 + len(args.model_names)

    # All values above 20 are going to be clipped to 20, as the deltaE00 values are usually below 20.

    if value is not None:
        for deltaE00_map in deltaE00_maps:
            deltaE00_map[deltaE00_map > value] = value
        # Set a maximum value of 20 for the colorbar, as the deltaE00 values are usually below 20.
        top_value = value
    else:
        max_value = max([map.max() for map in deltaE00_maps])
        top_value = np.ceil(max_value / 5) * 5
    norm = plt.Normalize(vmin=0, vmax=top_value)
    subplot_width = 5

    fig_width = subplot_width * cols
    fig_height = subplot_width * aspect_ratio * rows

    fig, axs = plt.subplots(2, 1 + len(args.model_names), figsize=(fig_width, fig_height))
    axs[0, 0].imshow(input_img)
    axs[0, 0].axis('off')
    axs[1, 0].imshow(gt_img)
    axs[1, 0].axis('off')
    for i, (pred_img, deltaE00_map) in enumerate(zip(pred_img_list, deltaE00_maps)):
        axs[0, i + 1].imshow(pred_img)
        axs[0, i + 1].axis('off')
        cax = axs[1, i + 1].imshow(deltaE00_map, norm=norm)
        axs[1, i + 1].axis('off')
    # No white space between rows
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
    # Add colorbar as an extra column at the end, with the height of one column (2 images on top of each other), and so
    # the colorbar is going to be shared between all the deltaE00 maps.
    # Add a title to the columns where the deltaE00 maps are plotted, leaving enough space between them and the images.
    axs[0, 0].set_title(f'Input {args.image_name}', pad=30)
    for i, model_name in enumerate(args.model_names):
        axs[0, i + 1].set_title(model_name, pad=30)
    plt.show()


if __name__ == '__main__':
    main()