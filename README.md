# proGAN
My own implementation of Progressive Growing of GANs, written in PyTorch. https://arxiv.org/abs/1710.10196

---------

This type of Generative Adversarial Network consists of training the generator and discriminator by gradually inserting new layers at runtime and thus increasing the resolution.  
The loss used in this GAN is the Wasserstein loss with gradient penalty.
