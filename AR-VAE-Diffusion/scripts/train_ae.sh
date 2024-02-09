# # # VAE_training Abstractart
python main/train_ae.py +dataset=abstractart/train \
                     dataset.vae.data.root=\'/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/images\' \
                     dataset.vae.data.name=\'abstractart\' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=32 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1000 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/\' \
                     dataset.vae.training.workers=16 \
                     dataset.vae.training.chkpt_prefix=\'second_training\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.delta=3.0 \
                     dataset.vae.training.gamma=10.0 

# # VAE_training for disentanglement measurements
# python main/train_ae.py +dataset=abstractart/train \
#                      dataset.vae.data.root=\'/home/anonymized/ASLR_DiffuseVAE/dataset/abstract-art/images\' \
#                      dataset.vae.data.name=\'abstractart\' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1000 \
#                      dataset.vae.training.device=\'gpu:0\' \
#                      dataset.vae.training.results_dir=\'/home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/\' \
#                      dataset.vae.training.workers=16 \
#                      dataset.vae.training.chkpt_prefix=\'noalsr_only_beta\' \
#                      dataset.vae.training.alpha=1.0 \
#                      dataset.vae.training.delta=3.0 \
#                      dataset.vae.training.gamma=0.0

