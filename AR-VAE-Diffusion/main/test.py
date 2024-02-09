import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import evaluation_metrics
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.latent import UncondLatentDataset
from models.vae import VAE
from util import configure_device, get_dataset, save_as_images


@click.group()
def cli():
    pass

def compute_eval_metrics(latent_codes, attributes, attr_list):
        """Function to tract disentanglement metrics. 
           From https://github.com/ashispati/ar-vae/blob/0d7f3513311e1056ed429ff53bdf57562d9e3a4a/measurevae/measure_vae_trainer.py
           Returns the saved results as dict or computes them"""
        
        interp_metrics = evaluation_metrics.compute_interpretability_metric(
            latent_codes, attributes, attr_list
        )
        metrics = {
            "interpretability": interp_metrics
        }
        metrics.update(evaluation_metrics.compute_correlation_score(latent_codes, attributes))
        metrics.update(evaluation_metrics.compute_modularity(latent_codes, attributes))
        metrics.update(evaluation_metrics.compute_mig(latent_codes, attributes))
        metrics.update(evaluation_metrics.compute_sap_score(latent_codes, attributes))
        return metrics

def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


# # TODO: Upgrade the commands in this script to use hydra config
# # and support Multi-GPU inference
# @cli.command()
# @click.argument("chkpt-path")
# @click.argument("root")
# @click.option("--device", default="gpu:1")
# @click.option("--dataset", default="celebamaskhq")
# @click.option("--image-size", default=128)
# @click.option("--num-samples", default=-1)
# @click.option("--save-path", default=os.getcwd())
# @click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
# def reconstruct(
#     chkpt_path,
#     root,
#     device="gpu:0",
#     dataset="celebamaskhq",
#     image_size=128,
#     num_samples=-1,
#     save_path=os.getcwd(),
#     write_mode="image",
# ):
#     dev, _ = configure_device(device)
#     if num_samples == 0:
#         raise ValueError(f"`--num-samples` can take value=-1 or > 0")

#     # Dataset
#     dataset = get_dataset(dataset, root, image_size, norm=False, flip=False)

#     # Loader
#     loader = DataLoader(
#         dataset,
#         16,
#         num_workers=4,
#         pin_memory=True,
#         shuffle=False,
#         drop_last=False,
#     )
#     vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(device)
#     vae.eval()

#     sample_list = []
#     img_list = []
#     count = 0
#     for _, batch in tqdm(enumerate(loader)):
#         batch = batch.to(dev)
#         with torch.no_grad():
#             recons = vae.forward_recons(batch)

#         if count + recons.size(0) >= num_samples and num_samples != -1:
#             img_list.append(batch[:num_samples, :, :, :].cpu())
#             sample_list.append(recons[:num_samples, :, :, :].cpu())
#             break

#         # Not transferring to CPU leads to memory overflow in GPU!
#         sample_list.append(recons.cpu())
#         img_list.append(batch.cpu())
#         count += recons.size(0)

#     cat_img = torch.cat(img_list, dim=0)
#     cat_sample = torch.cat(sample_list, dim=0)

#     # Save the image and reconstructions as numpy arrays
#     os.makedirs(save_path, exist_ok=True)

#     if write_mode == "image":
#         save_as_images(
#             cat_sample,
#             file_name=os.path.join(save_path, "vae"),
#             denorm=False,
#         )
#         save_as_images(
#             cat_img,
#             file_name=os.path.join(save_path, "orig"),
#             denorm=False,
#         )
#     else:
#         np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
#         np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
):
    seed_everything(seed)
    dev, _ = configure_device(device)

    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")
    dev="cuda"
    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1))

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())

@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
@click.option("--attribute-variation", default="[5,0]", type=str)
@click.option("--attribute-index", default="[0,1]", type=str)
def sample_with_attributes(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
    attribute_variation="[5,5]",
    attribute_index="[0,1]"
):
    seed_everything(seed)
    dev, _ = configure_device(device)

    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")
    dev="cuda"
    attribute_variation = eval(attribute_variation)
    attribute_index = eval(attribute_index)
    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1), add_variation=attribute_variation, attribute_index=attribute_index, thrice=True)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= 3*num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample[:num_samples],
            file_name=os.path.join(save_path+"_neg", "vae"),
            denorm=False,
        )
        save_as_images(
            cat_sample[num_samples:2*num_samples],
            file_name=os.path.join(save_path+"_neutral", "vae"),
            denorm=False,
        )
        save_as_images(
            cat_sample[2*num_samples:3*num_samples],
            file_name=os.path.join(save_path+"_pos", "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())

@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
@click.option("--attribute-variation", default="[5,0]", type=str)
@click.option("--attribute-index", default="[0,1]", type=str)
def sample_one_range(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
    attribute_variation="[5,5]",
    attribute_index="[0,1]"
):
    seed_everything(seed)
    dev, _ = configure_device(device)

    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")
    dev="cuda"
    attribute_variation = eval(attribute_variation)
    attribute_index = eval(attribute_index)
    dataset = UncondLatentDataset((1, z_dim, 1, 1), add_variation=attribute_variation, attribute_index=attribute_index, one=True, thrice=False)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= 36*num_samples and num_samples != -1:
            sample_list.append(recons[:36*num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path+f"variation{str(attribute_variation)}", "vae"),
            denorm=False,
            grid=True,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())
@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=128)
@click.option("--attribute_index", default='[0,1]', type=str)
@click.option("--attributes-data", default="/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/attribute_data.csv")
@click.option("--dataset-name", default="abstractart")
@click.option("--root", default="/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/images")
@click.option("--save-path", default=os.getcwd())
def compute_disentanglement_metrics(
    z_dim,
    chkpt_path,
    seed=0,
    root="/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/images",
    device="gpu:0",
    image_size=128,
    attribute_index='[0,1]',
    dataset_name="abstractart",
    attributes_data = "/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/attribute_data.csv",
    save_path=os.getcwd(),
):
    seed_everything(seed)
    dev, _ = configure_device(device)
    dev="cuda"
    device = dev
    attribute_index = eval(attribute_index)
    # Dataset
    dataset = get_dataset(dataset_name, root, image_size, attribute_data=attributes_data, norm=False, flip=False)
    # Loader
    # Define the percentage of data you want to load (20%)
    percentage = 0.2
    total_data = len(dataset)
    subset_size = int(percentage * total_data)

    # Create a Subset dataset with the first 20% of the data
    subset_dataset = Subset(dataset, range(subset_size))
    loader_dataset = DataLoader(
        subset_dataset,
        1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(device)
    vae.eval()
    latent_codes = []
    attr_val_dict = {key: [] for key in attribute_index}
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader_dataset)):
            data, attribute_values = batch
            data = data.to(device)
            mu, logvar = vae.encode(data)
            z = vae.reparameterize(mu, logvar)
            squeezed_z = torch.squeeze(z)
            latent_codes.append(squeezed_z.cpu().numpy())
            for index in attribute_index:
                    attr_val = attribute_values[:,index]
                    attr_val_dict[index].extend(attr_val.tolist())
        attributes_for_eval = np.array(list(zip(*[attr_val_dict[index] for index in attribute_index])))
        latent_codes = np.array(latent_codes)
        latent_codes = latent_codes
        print(latent_codes.shape, attributes_for_eval.shape)
        print(list(loader_dataset)[0])
        print("\nNext\n")
        print(attributes_for_eval[0, :])
        with open(save_path + "disentanglement_metrics.txt", 'a') as file:
            results_metric = compute_eval_metrics(np.array(latent_codes), attributes_for_eval, attribute_index)
            file.write("%s\t" % (chkpt_path))
            for key, value in results_metric.items(): 
                file.write('%s:%s\t' % (key, value))
            file.write("\n")



   


if __name__ == "__main__":
    cli()


