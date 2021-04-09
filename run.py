import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from skimage import io
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from matplotlib import pyplot as plt

from PIL import Image

from progan.modeling import GANConfiguration
from progan.loss import GANLoss
from progan.trainer import TrainingArguments, GANTrainer

LEARNING_RATE = 1e-3
LAMBDA_GP = 10

if __name__ =="__main__":

    transform = transforms.Compose([
            transforms.Scale((128, 128)),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),

        ])

    celeba = torchvision.datasets.ImageFolder(root="./img_align_celeba", transform = transform)

    gan_config = GANConfiguration()

    gan_config.save("configs/default_gan")

    generator, discriminator = gan_config.build_gan()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator.to(device)
    discriminator.to(device)

    print(f"Device : {device}")

    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0, 0.99))
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0, 0.99))

    gan_loss = GANLoss(g_optimizer, d_optimizer, LAMBDA_GP)
    training_args = TrainingArguments(save_steps = 1000, checkpoint_imgs=32, num_workers=2)
    trainer = GANTrainer(discriminator, generator, gan_loss, celeba, training_args)

    trainer.train()