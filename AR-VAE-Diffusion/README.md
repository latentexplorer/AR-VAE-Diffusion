# Codebase for AR-VAE Diffusion models

This repo contains the official implementation of the attached paper. The code in this repository was initially forked from the DiffuseVAE paper repository (https://github.com/kpandey008/DiffuseVAE) and then modified.

## Setup
We used conda as our package manager. The assocaited environment.yml contains the dependencies of this project. An enviroment can be created with
```
conda env create -f environment.yml
```
## Hyperparameters
We use hydra to manage hyperparameter values. Config examples are available in main/configs/dataset for each dataset.

## Scripts
There are 4 bash scripts. Two for training and testing the AR-VAE and two for training and testing the DDPM. Each of these can be run with
```
bash scripts/script.sh
```
## Important Parameters for generating images.
To generate images using this repository the following parameters have to be configured.
| Parameter | Description |
| --- | --- |
| ***Data and Checkpoint Paths*** | These need to be set as per the location of the resources in the local machine |
| *scripts/test_ddpm.sh* | |
| `dataset.ddpm.evaluation.eval_mode` | Choose between generation, vae variations, ddpm variations, reconstructions, and alsr varied generations. Has to be one of ["sample", "recons", "alsr", "variations_ddpm", "variations_vae"] | 
| `dataset.vae.evaluation.chkpt_path` | Checkpoint of the VAE used for evaluation |
| `dataset.ddpm.evaluation.chkpt_path` | Checkpoint path for the Diffusion model used for evalution |
| `+dataset=abstractart/test` | Specifies the dataset config used for evaluation |
| *main/configs/dataset/curlnoise/test.yaml* | |
| `ddpm: data: root:` | The location of the images of the dataset used in different evaluations (for example, reconstruction) | 
| | |
| ***Controllability Related parameters*** | Used to control generations |
| *scripts/test_ddpm.sh* | | 
| `dataset.ddpm.evaluation.add_variation` | Amount of DELTA to add for variation along a specific attribute. Can be a float. |
| `dataset.ddpm.evaluation.attribute_index` | Used to specify the latent index of the attribute the variation has to be added to.|
| `dataset.ddpm.evaluation.normal_mean` | Change mean of normal distribution added (to make images more interesting) to latent sample before generation|
| `dataset.ddpm.evaluation.normal_std` |  Change standard deviation of normal distribution added (to make images more interesting) to latent sample before generation|
| | |
| ***Miscellaneous parameters related to image generation*** | These parameters significantly affect how the images are generated | 
| *scripts/test_ddpm.sh* | |
| `dataset.ddpm.evaluation.n_sample` | Number of samples to generate |
| `dataset.ddpm.evaluation.batch_size=4` | Batch size for generation. Bigger batches require more memory |
| `dataset.ddpm.evaluation.sample_method` | Sampling method for the Diffusion Model. Can be ddpm or ddim |
| `dataset.ddpm.evaluation.n_steps` | Number of sampling steps used to sample. |
| `dataset.ddpm.evaluation.skip_strategy'` | Sampling strategy for DDIM. Can be uniform or quad. |
| `dataset.ddpm.evaluation.resample_strategy` | Support for truncated sampling. For DDIM set to spaced. |

## AR-VAE related testing
The VAE generations and metrics and be tested with the ```test_ae.sh``` script.
There are 3 main functionalities offered using clicky:
- sample: sample images from the vae
- sample-with-attributes: sample images while adding variation
- compute-disentanglement-metrics: computed the disentanglement metrics
Example configurations are available in the bash file with parameter configurations

## License
The given License is the exact same MIT License from the DiffuseVAE paper repository and the author information in it does not relate to the authors of this paper.