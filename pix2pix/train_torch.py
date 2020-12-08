import os
import logging
from math import log10

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split

from data_load_torch import DatasetLoader
from network import Generator, Discriminator
from loss_torch import GANLoss

input_dir = "/home/sonsungwoo/pix2pix/data/image/"
label_dir = "/home/sonsungwoo/pix2pix/data/label/"
save_path = "/home/sonsungwoo/pix2pix/result/"
image_size = 1024
val_percent = 0.1
batch_size = 10
epochs = 1000
lr = 0.001
save_cp = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta1 = 0.5
beta2 = 0.999
lambda_As = 100.0
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def train():
    print("data loading")
    dataset = DatasetLoader(input_dir, label_dir, image_size)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {image_size}
    """
    )

    # Networks
    generator = Generator(batch_size)
    discriminator = Discriminator(batch_size)

    # checkpoint = torch.load("./checkpoint/model_epoch_200.pth")

    generator = nn.DataParallel(generator).to(device=device)
    discriminator = nn.DataParallel(discriminator).to(device=device)

    # generator.load_state_dict(checkpoint["g_state_dict"])
    # discriminator.load_state_dict(checkpoint["d_state_dict"])

    # generator.load_state_dict(g_checkpoint)
    generator.train()

    # discriminator.load_state_dict(d_checkpoint)
    discriminator.train()

    # Losses
    criterionGAN = GANLoss().to(device)

    criterionL1 = nn.L1Loss().to(device)

    criterionMSE = nn.MSELoss().to(device)


    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr, [beta1, beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), lr, [beta1, beta2])

    # g_optimizer.load_state_dict(checkpoint["g_optimizer_dict"])
    # d_optimizer.load_state_dict(checkpoint["d_optimizer_dict"])

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        train_epoch = epoch 
        print("epoch: ", train_epoch)
        for sample in train_loader:
            input_image = sample["input_image"].to(device=device)
            label_image = sample["label_image"].to(device=device)

            # discriminator
            d_optimizer.zero_grad()

            fake_image = generator(input_image)

            pred_fake = discriminator(input_image, fake_image)
            loss_D_fake = criterionGAN(pred_fake, False)

            pred_real = discriminator(input_image, label_image)
            loss_D_real = criterionGAN(pred_real, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            d_optimizer.step()

            # generator
            g_optimizer.zero_grad()

            fake_image = generator(input_image)

            pred_real = discriminator(input_image, fake_image)
            loss_G_gan = criterionGAN(pred_real, True)

            
            loss_G_l1 = criterionL1(fake_image, label_image) * lambda_As

            loss_G = loss_G_gan + loss_G_l1

            loss_G.backward()
            g_optimizer.step()

        avg_psnr = 0

        num = 0

        if epoch % 50 == 0:
            for sample in val_loader:
                input_image = sample["input_image"].to(device=device)
                label_image = sample["label_image"].to(device=device)

                fake_image = generator(input_image)
                mse = criterionMSE(fake_image, label_image)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_loader)))

                if num < 2:
                    torchvision.utils.save_image(
                        denorm(fake_image),
                        os.path.join(
                            save_path, "Fake image-%d-%d.tif" % (train_epoch, num + 1)
                        ),
                    )
                    if epoch == 0:
                        torchvision.utils.save_image(
                            denorm(input_image),
                            os.path.join(
                                save_path,
                                "Input image-%d-%d.tif" % (train_epoch, num + 1),
                            ),
                        )
                        torchvision.utils.save_image(
                            denorm(label_image),
                            os.path.join(
                                save_path,
                                "Label image-%d-%d.tif" % (train_epoch, num + 1),
                            ),
                        )
                    num += 1

            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            model_out_path = "checkpoint/model_epoch_{}.pth".format(train_epoch)

            torch.save(
                {
                    "g_state_dict": generator.state_dict(),
                    "d_state_dict": discriminator.state_dict(),
                    "g_optimizer_dict": g_optimizer.state_dict(),
                    "d_optimizer_dict": d_optimizer.state_dict(),
                },
                model_out_path,
            )

            print("Checkpoint saved to {}".format("checkpoint"))


if __name__ == "__main__":
    train()
