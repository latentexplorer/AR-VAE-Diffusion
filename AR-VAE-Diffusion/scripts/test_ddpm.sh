python main/eval/ddpm/sample_cond.py +dataset=abstractart/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dim_mults=\'1,1,2,2,4,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.eval_mode="sample" \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'path\' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='uniform' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1 \
                        dataset.ddpm.evaluation.batch_size=4 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=50 \
                        dataset.ddpm.evaluation.n_steps=200 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=18 \
                        dataset.ddpm.evaluation.normal_mean=0 \
                        dataset.ddpm.evaluation.normal_std=1 \
                        dataset.ddpm.evaluation.add_variation=0 \
                        dataset.ddpm.evaluation.attribute_index=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/home/anonymized/ASLR_DiffuseVAE/results/abstractart_ddpm/ddpmv2-VAE1000_abstractart-epoch=999-loss=0.0073.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/vae-second_training-epoch=999-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/anonymized/ASLR_DiffuseVAE/results/variations/completed_vae_2\' \


