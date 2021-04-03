import torch

class GANLoss:
    def __init__(self, g_optimizer, d_optimizer, gradient_lambda):
        self.gradient_lambda = gradient_lambda
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
    def gradient_penalty(self, discriminator, real, fake, alpha):
        batch_size, c, h, w = real.shape
        beta = torch.rand((batch_size, 1, 1, 1), device = discriminator.get_device())
        interpolated_images = real * beta + fake * (1 - beta)

        # Calculate critic scores
        mixed_scores = discriminator(interpolated_images, alpha)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty

    
    def g_loss_optimize(self, discriminator, real, fake, alpha=1):
        g_res = discriminator(fake, alpha).view(-1)
        g_loss = -torch.mean(g_res)
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss
    
    def d_loss_optimize(self, discriminator, real, fake, alpha=1):
        rd_res = discriminator(real, alpha).view(-1)
        fd_res = discriminator(fake, alpha).view(-1)

        gp = self.gradient_penalty(discriminator, real, fake, alpha) 
        d_loss = torch.mean(fd_res)-torch.mean(rd_res) + self.gradient_lambda * gp
        
        self.d_optimizer.zero_grad()
        d_loss.backward(retain_graph = True)
        self.d_optimizer.step()
        
        return d_loss