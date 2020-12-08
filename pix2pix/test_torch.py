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

input_dir = "./test/cell/"
label_dir = "./test/label/"
save_path = "./test_result/"
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

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {image_size}
    """
    )

    # Networks
    generator = Generator(batch_size)
    discriminator = Discriminator(batch_size)

    checkpoint = torch.load("./checkpoint/model_epoch_150.pth")

    generator = nn.DataParallel(generator).to(device=device)
    discriminator = nn.DataParallel(discriminator).to(device=device)

    generator.load_state_dict(checkpoint["g_state_dict"])
    discriminator.load_state_dict(checkpoint["d_state_dict"])

    generator.eval()
    discriminator.eval()

    criterion = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    for epoch in range(1):
        train_epoch = epoch
        print("epoch: ", train_epoch)

        avg_loss = 0
        avg_psnr = 0

        for sample in train_loader:
            input_image = sample["input_image"].to(device=device)
            label_image = sample["label_image"].to(device=device)

            # discriminator

            fake_image = generator(input_image)
            loss = criterion(fake_image, label_image)
            mse = criterionMSE(fake_image, label_image)
            psnr = 10 * log10(1 / mse.item())

            avg_loss += loss.item()
            
            
            avg_psnr += psnr

            

            torchvision.utils.save_image(
                denorm(fake_image),
                os.path.join(save_path, "Fake image-%d-%d.tif" % (train_epoch + 1, 1),),
            )
            torchvision.utils.save_image(
                denorm(input_image),
                os.path.join(
                    save_path, "Input image-%d-%d.tif" % (train_epoch + 1, 1),
                ),
            )
            torchvision.utils.save_image(
                denorm(label_image),
                os.path.join(
                    save_path, "Label image-%d-%d.tif" % (train_epoch + 1, 1),
                ),
            )
        
        print("===> Avg. loss: {:.4f} ".format(avg_loss / len(train_loader)))    
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(train_loader)))


if __name__ == "__main__":
    train()
