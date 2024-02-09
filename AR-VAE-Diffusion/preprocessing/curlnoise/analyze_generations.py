"""
This file is used to analyze and plot various plots for a trained model. Current functionality includes functions like:
1. generate_around_sample: See how decoded images change when noise (normal) is added to their latent space encodings.
2. generated: generate images. Can be used to test different ways to sample from latent space
3. prototypes_distance: Check and plot training examples along with the prototype they are the closest to
4. tsne: tsne plots of a subset of the encoded dataset and the prototypes
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms
import train
from sklearn.metrics.pairwise import euclidean_distances
from  VAE import VariationalAutoencoder
import utils
import random
from datasetLoader import CurlNoiseDataset
import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

batch_size =  16 
gpu = torch.device("cuda")
device = torch.device("cpu")
print(f'Selected device: {device}')

enc_block_config_str2 = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
enc_channel_config_str2 = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
dec_block_config_str2 = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
dec_channel_config_str2 = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
vae = VariationalAutoencoder(
        128,
        enc_block_config_str2,
        dec_block_config_str2,
        enc_channel_config_str2,
        dec_channel_config_str2,
        1024
)

checkpoint = torch.load(
    "/home/anonymizedsood/Desktop/VAE_latent_regularisation/VAEL-SR/models/2023-02-13/16-37-43 Models Comparing 2 forms of latent regularisation 200 16 1024 0.0001 10 1 1 0/checkpoints/200.pt",
    map_location=device)
vae.load_state_dict(checkpoint["model_state_dict"])

data_dir_name = "../data/retained/imagefolder128/train"
train_dataset = CurlNoiseDataset(data_dir_name)
train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
# apply transform
train_dataset.transform = train_transform
n = len(train_dataset)
print(n)
train_data, val_data = random_split(train_dataset, [round(n - n * 0.2), round(n * 0.2)])

# load data
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# Generate images, copied from train
def generated(vae):
    vae.to(device)
    with torch.no_grad():
        # sample latent vectors from the normal distribution
        latent = torch.randn(128, vae.latent_dims, device=device)       
        latent = latent[:,:,None,None]
        
        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        print((img_recon.data[:100]).shape)
        img = train.show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
        plt.show()

def compute_eval_metrics(latent_codes, attributes, attr_list):
        """Function to tract disentanglement metrics. 
           From https://github.com/ashispati/ar-vae/blob/0d7f3513311e1056ed429ff53bdf57562d9e3a4a/measurevae/measure_vae_trainer.py
           Returns the saved results as dict or computes them"""
        
        interp_metrics = utils.compute_interpretability_metric(
            latent_codes, attributes, attr_list
        )
        metrics = {
            "interpretability": interp_metrics
        }
        metrics.update(utils.compute_correlation_score(latent_codes, attributes))
        metrics.update(utils.compute_modularity(latent_codes, attributes))
        metrics.update(utils.compute_mig(latent_codes, attributes))
        metrics.update(utils.compute_sap_score(latent_codes, attributes))
        return metrics

attributes = ["pixel_density", "size"]
def test_metrics(vae, device, dataloader, path, checkpoints, filename):
    vae.eval()
    for checkpoint in checkpoints:
        load_chpkt = torch.load(
            path + f"{checkpoint}.pt")
        vae.load_state_dict(load_chpkt["model_state_dict"])
        vae.to(device)
        latent_codes = []
        attr_val_dict = {key: [] for key in attributes}
        with torch.no_grad():
            for x in tqdm.tqdm(dataloader):
                data = x["data"]
                data = data.to(device)
                z, x_hat = vae(data)
                squeezed_z = torch.squeeze(z)
                latent_codes.extend(squeezed_z.cpu().numpy())
                for attribute in attributes:
                        attr_val = x[attribute].to(device)
                        attr_val_dict[attribute].extend(attr_val.tolist())

            attributes_for_eval = np.array(list(zip(*[attr_val_dict[attribute] for attribute in attributes])))
            latent_codes = np.array(latent_codes)
            latent_codes = latent_codes
            
            with open(filename, 'a') as file:
                results_metric = compute_eval_metrics(np.array(latent_codes), attributes_for_eval, attributes)
                file.write("%s\t" % (checkpoint))
                for key, value in results_metric.items(): 
                    file.write('%s:%s\t' % (key, value))
                file.write("\n")
path = "/home/anonymizedsood/Desktop/VAE_latent_regularisation/VAEL-SR/models/2023-02-13/16-37-43 Models Comparing 2 forms of latent regularisation 200 16 1024 0.0001 10 1 1 0/checkpoints/"
path2 = "/home/anonymizedsood/Desktop/VAE_latent_regularisation/VAEL-SR/models/2023-02-17/18-04-28 Models Comparing 2 forms of latent regularisation 200 16 1024 0.0001 10 1 1 0/checkpoints/"
test_metrics(vae, gpu, valid_loader, path, [5, 50, 100, 150, 200], "AR-Metrics-2.txt")
test_metrics(vae, gpu, valid_loader, path2, [5, 50, 100, 150, 200], "Beta-Metrics-2.txt")
generated(vae)