python main/test.py sample --device gpu:0 \
                            --image-size 128 \
                            --seed 0 \
                            --num-samples 32 \
                            --save-path /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/test_outputs \
                            --write-mode image \
                            1024 \
			    /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/vae-first_training-epoch=365-train_loss=0.0000.ckpt \

# python main/test.py compute-disentanglement-metrics --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --save-path /home/anonymized/ASLR_DiffuseVAE/results/disentanglement_data_testing \
#                             1024 \
# 			                /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/checkpoints/vae-second_training-epoch=999-train_loss=0.0000.ckpt

# python main/test.py sample-with-attributes --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 16 \
#                             --save-path /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/test_outputs_color_variaton \
#                             --write-mode image \
#                             --attribute-variation [0,50] \
#                             --attribute-index [0,1] \
#                             1024 \
# 			                /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/vae-second_training-epoch=999-train_loss=0.0000.ckpt \
# python main/test.py sample-one-range --device gpu:0 \
#                             --image-size 128 \
#                             --seed 153 \
#                             --save-path /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/test_outputs_structure_variation \
#                             --write-mode image \
#                             --attribute-variation [0,5] \
#                             --attribute-index [0,1] \
#                             1024 \
# 			                /home/anonymized/ASLR_DiffuseVAE/results/abstractart_vae/vae-second_training-epoch=999-train_loss=0.0000.ckpt \
