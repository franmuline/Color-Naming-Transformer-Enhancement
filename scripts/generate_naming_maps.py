"""
Visualize the color naming maps for a given image. It shows the color naming probabilities for each pixel in the image.
It also shows them color coded, so the visualization is more intuitive.

Code provided by:
David Serrano (dserrano@cvc.uab.cat)
"""

import torch
import argparse
import os.path
from PIL import Image
from basicsr.data.color_naming.color_naming import ColorNaming
from torchvision.transforms import functional as TF


color_map_names_6 = ['orange-brown-yellow', 'achromatic', 'pink-purple', 'red', 'green', 'blue']
color_map_names_11 = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_categories', type=int, default=6)
    parser.add_argument('--image_path', type=str, default='../../Datasets/FiveK/input/a4076-_DGW6244.png')
    parser.add_argument('--output_directory', type=str, default='./color_naming_maps')
    parser.add_argument('--save_individual', type=bool, default=True)
    parser.add_argument('--plot_cn_probs', type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    color_naming = ColorNaming(num_categories=args.num_categories)

    if os.path.isfile(args.image_path):
        image_tensor = TF.pil_to_tensor(Image.open(args.image_path).convert('RGB')).unsqueeze(0)

        cn_probs = color_naming(image_tensor / 255.0).float().repeat(1, 3, 1, 1).cpu()
        output_images = (1 - cn_probs) * 255 * torch.ones_like(image_tensor).repeat(args.num_categories, 1, 1,
                                                                                    1) + cn_probs * image_tensor.repeat(
            args.num_categories, 1, 1, 1)
        if args.plot_cn_probs:
            if not args.save_individual:
                import matplotlib.pyplot as plt
                fig = plt.subplots(1, args.num_categories, figsize=(20, 20))
                for i in range(args.num_categories):
                    plt.subplot(1, args.num_categories, i+1)
                    plt.imshow(cn_probs[i].permute(1, 2, 0).numpy())
                plt.show()
            else:
                # Get the image name
                image_name = os.path.basename(args.image_path).split('.')[0]
                os.makedirs(args.output_directory + f'/{image_name}/cn_probs/', exist_ok=True)
                for i in range(args.num_categories):
                    # Transform tensor to PIL image
                    numpy_image = cn_probs[i].permute(1, 2, 0).numpy()
                    img_to_save = Image.fromarray((numpy_image * 255).astype('uint8'))
                    img_to_save.save(args.output_directory + f'/{image_name}/cn_probs/{color_map_names_6[i] if args.num_categories == 6 else color_map_names_11[i]}.png')


        if not args.save_individual:
            import matplotlib.pyplot as plt

            fig = plt.subplots(1, args.num_categories, figsize=(20, 20))
            for i in range(args.num_categories):
                plt.subplot(1, args.num_categories, i + 1)
                # plt.imsave(f'map_{i}.png', output_images[i].permute(1, 2, 0).numpy().astype('uint8'))
                plt.imshow(output_images[i].permute(1, 2, 0).numpy().astype('uint8'))
            plt.show()
        else:
            # Get the image name
            image_name = os.path.basename(args.image_path).split('.')[0]
            os.makedirs(args.output_directory + f'/{image_name}/color_coded/', exist_ok=True)
            for i in range(args.num_categories):
                # Transform tensor to PIL image
                img_to_save = TF.to_pil_image(output_images[i].byte())
                img_to_save.save(args.output_directory + f'/{image_name}/color_coded/{color_map_names_6[i] if args.num_categories == 6 else color_map_names_11[i]}.png')