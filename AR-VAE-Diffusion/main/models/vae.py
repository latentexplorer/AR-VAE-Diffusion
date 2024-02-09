import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class Encoder(nn.Module):
    def __init__(self, block_config_str, channel_config_str):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, down_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue
            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)

        # Latents
        self.mu = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)
        self.logvar = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)

    def forward(self, input):
        x = self.in_conv(input)
        x = self.block_mod(x)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str):
        super().__init__()
        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)
        self.last_conv = nn.Conv2d(channel_config[input_res], 3, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.block_mod(input)
        x = self.last_conv(x)
        return torch.sigmoid(x)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):
    def __init__(
        self,
        max_epochs,
        input_res,
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        attributes,
        alpha_start=0,
        alpha_stop=1,
        n_cycle=4,
        ratio=0.5,
        gamma=5,
        delta=1,
        lr=1e-4,
        alpha=1,
    ):
        super().__init__()

        def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
            L = np.ones(n_epoch)
            period = n_epoch/n_cycle
            step = (stop-start)/(period*ratio) # linear schedule

            for c in range(n_cycle):

                v , i = start , 0
                while v <= stop and (int(i+c*period) < n_epoch):
                    L[int(i+c*period)] = v
                    v += step
                    i += 1
            return L

        self.save_hyperparameters()
        self.input_res = input_res
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str
        self.max_epochs = max_epochs
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.alpha = alpha
        # self.alphas = frange_cycle_linear(self.alpha_start, self.alpha_stop, self.max_epochs, self.n_cycle, self.ratio)
        self.attributes = attributes.split(",")
        # self.attribute_data = pd.read_csv(attribute_data)
        # self.attr_vals = 
        self.lr = lr
        self.delta = delta
        self.gamma = gamma

        # Encoder architecture
        self.enc = Encoder(self.enc_block_str, self.enc_channel_str)

        # Decoder Architecture
        self.dec = Decoder(self.input_res, self.dec_block_str, self.dec_channel_str)
    
    def encode(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reg_loss_sign(self, latent_code, attribute, delta=1.0):
        """
            Computes the regularization loss given the latent code and attribute
            Args:
                latent_code: torch Variable, (N,)
                attribute: torch Variable, (N,)
                factor: parameter for scaling the loss
            Returns
                scalar, loss
            Credit: https://github.com/ashispati/ar-vae/blob/0d7f3513311e1056ed429ff53bdf57562d9e3a4a/utils/trainer.py#L370
        """
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * delta)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())
        return sign_loss

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z)

        # Compute loss
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(decoder_out, x)
        kl_loss = self.compute_kl(mu, logvar)
        ar_loss = 0
        j = 0
        for attribute in self.attributes: 
            attr_val = batch[1][:,j]
            ar_loss += self.reg_loss_sign(z[:, j], attr_val, self.delta)
            j += 1 
                
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)
        self.log("Attribute Loss", ar_loss, prog_bar=True)
        total_loss = recons_loss + self.alpha * kl_loss + self.gamma * ar_loss
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    vae = VAE(
        1000,
        128,
        enc_block_config_str,
        dec_block_config_str,
        enc_channel_config_str,
        dec_channel_config_str,
        "structural_complexity,color_diversity",        
    )
    print(vae)

