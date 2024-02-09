import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
from datasets import (
    CurlNoise,
    AbstractArt
)

logger = logging.getLogger(__name__)


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = -1
        gpu_id = [int(id) for id in gpu_id.split(",")]
        return f"cuda:{gpu_id}", gpu_id
    return device


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :return: a set of diffusion steps from the original process to use.
    """
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(
            f"cannot create exactly {desired_count} steps with an integer stride"
        )
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError


def get_dataset(name, root, image_size, attribute_data=None, norm=True, flip=False, **kwargs):
    assert isinstance(norm, bool)

    # Construct transforms
    t_list = [T.Resize((image_size, image_size))]
    if flip:
        t_list.append(T.RandomHorizontalFlip(0.25))
        t_list.append(T.RandomVerticalFlip(0.25))
    transform = T.Compose(t_list)

    if name == "curlnoise":
        dataset = CurlNoise(root, attribute_data=attribute_data, norm=norm, transform=transform, **kwargs)
    elif name == "abstractart":
        dataset = AbstractArt(root, attribute_data=attribute_data, norm=norm, transform=transform, **kwargs)
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.squeeze().permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1,
    interpolations_2,
    save_path=None,
    figsize=(10, 2),
    denorm=True,
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        # De-Norm
        inter_1 = 0.5 * inter_1 + 0.5 if denorm else inter_1
        # inter_2 = 0.5 * inter_2 + 0.5 if denorm else inter_2

        # Plot
        ax[0, i].imshow(inter_1.squeeze().permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.squeeze().permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)


def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(3):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (
            channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0]
        )
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj

# Function to create a grid of images
def create_image_grid(images, grid_size):
    # Calculate the size of the final grid image
    grid_width = max(image.width for image in images) * grid_size[1]
    grid_height = max(image.height for image in images) * grid_size[0]

    # Create a new blank image with the calculated size
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Paste each image onto the grid
    for i, image in enumerate(images):
        row = i // grid_size[1]
        col = i % grid_size[1]

        # Calculate the position to paste the current image
        paste_x = col * image.width
        paste_y = row * image.height

        # Paste the image onto the grid
        grid_image.paste(image, (paste_x, paste_y))

    return grid_image

def save_as_images(obj, file_name="output", denorm=True, grid=False):
    # Saves predictions as png images (useful for Sample generation)
    if denorm:
        # obj = normalize(obj)
        obj = obj * 0.5 + 0.5
    obj_list = convert_to_np(obj)
    
    if grid:

        images = []

        for i, out in enumerate(obj_list):
            out = (out * 255).clip(0, 255).astype(np.uint8)
            img_out = Image.fromarray(out)
            images.append(img_out)

        current_file_name = file_name + ".png"
        grid_image = create_image_grid(images, (6, 6))
        directory_path, _ = os.path.split(current_file_name)
        if not os.path.exists(directory_path):
            # If not, create the directory
            os.makedirs(directory_path)
        grid_image.save(current_file_name, "png")

    else:
        for i, out in enumerate(obj_list):
            out = (out * 255).clip(0, 255).astype(np.uint8)
            img_out = Image.fromarray(out)
            current_file_name = file_name + "_%d.png" % i
            directory_path, _ = os.path.split(current_file_name)
            if not os.path.exists(directory_path):
                # If not, create the directory
                os.makedirs(directory_path)
            img_out.save(current_file_name, "png")
        


def save_as_np(obj, file_name="output", denorm=True):
    # Saves predictions directly as numpy arrays
    if denorm:
        obj = normalize(obj)
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        current_file_name = file_name + "_%d.npy" % i
        np.save(current_file_name, out)


def compare_samples(samples, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    ncols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for idx, (caption, img) in enumerate(samples.items()):
        ax[idx].imshow(img.permute(1, 2, 0))
        ax[idx].set_title(caption)
        ax[idx].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)

    plt.close()
