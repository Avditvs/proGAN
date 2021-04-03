import time
import os
import shutil
import random
import json

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np

class TrainingArguments:
    def __init__(self,
                 save_steps=4000,
                 log_steps = 10,
                 log_dir="logs",
                 generate_steps=500,
                 rank_steps = [4000, 8000, 16000, 32000, 64000, 128000],
                 transition_steps = [4000, 8000, 16000, 32000, 64000],
                 batch_size = [16, 16, 16, 16, 16, 16],
                 num_workers = 0,
                 checkpoint_imgs = 8,
                 resume_from = None,
                 save_dir = "saves",
                 num_saves = 3
                ):
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.generate_steps = generate_steps
        self.log_dir = log_dir
        self.rank_steps = rank_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transition_steps = transition_steps
        self.checkpoint_imgs = checkpoint_imgs
        self.resume_from = resume_from
        self.save_dir = save_dir
        self.num_saves = num_saves


class GANTrainer:
    
    def __init__(self, discriminator, generator, gan_loss, dataset, args):
        self.discriminator = discriminator
        self.generator = generator
        self.gan_loss = gan_loss
        self.dataset = dataset
        self.args = args
        
        self.training_id = str(round(time.time()))
        
        self.n_iter = 0
        self.steps_in_rank = 0
        self.current_batch_size = self.args.batch_size[0]
        
        self.history = {'G_loss':[], 'D_loss':[], 'img_checkpoint':[], 'd_accuracy_fake':[], 'd_accuracy_real':[]}
                               
        try:
            shutil.rmtree(os.path.join(args.log_dir, self.training_id))
        except:
            pass
            
        self.writer = SummaryWriter(log_dir = os.path.join(args.log_dir, self.training_id))
        
        self.new_rank_steps = np.cumsum(self.args.rank_steps)

        if self.args.resume_from is not None:
            self.load(self.args.resume_from)
        
    
    def train(self):
        checkpoint_noise = self.generator.generate_latent_points(self.args.checkpoint_imgs)
        self.dataloader= DataLoader(self.dataset,
                                                 batch_size=self.current_batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers
                                                )
        
        self.iterator = iter(self.dataloader)
        while self.n_iter < self.new_rank_steps[-1]:
            
            factor = 2**(self.generator.depth-self.generator.rank)
            alpha = 1. if self.generator.rank==0 else min(self.steps_in_rank,
                                                          self.args.transition_steps[self.generator.rank-1]
                                                         )/self.args.transition_steps[self.generator.rank-1]
            
            try:
                real = next(self.iterator)
            except StopIteration:
                #del self.iterator
                self.iterator = iter(self.dataloader)
                real = next(self.iterator)
            
            n_items = len(real)
            real = real.to(self.discriminator.get_device())
            real = F.avg_pool2d(real, factor)   
            
            noise = self.generator.generate_latent_points(n_items)
            fake = self.generator(noise, alpha)
            
            d_loss = self.gan_loss.d_loss_optimize(self.discriminator, real, fake, alpha)

            g_loss = self.gan_loss.g_loss_optimize(self.discriminator, real, fake, alpha)
            
            self.log(d_loss, g_loss, checkpoint_noise, alpha)
            self.step()
            
        
        
    def log(self, d_loss, g_loss, checkpoint_noise, alpha):
        if self.n_iter%self.args.generate_steps==0:
            checkpoint_image = self.generate_checkpoint_images(checkpoint_noise, alpha)
            self.writer.add_image('generated', ((checkpoint_image+1)/2), self.n_iter)
                               
        if self.n_iter%self.args.log_steps==0:
            self.writer.add_scalar('Loss/Discriminator', - d_loss.item(), self.n_iter)
            self.writer.add_scalar('Loss/Generator', g_loss.item(), self.n_iter)
            
        if self.n_iter%self.args.save_steps==0:
            self.save()
            
    def generate_checkpoint_images(self, checkpoint_noise, alpha):
        with torch.no_grad():
            image = torchvision.utils.make_grid(self.generator(checkpoint_noise, alpha).detach().cpu().clamp(-1, 1), nrow=8)
        return image
            
    def step(self):
        self.n_iter +=1
        self.steps_in_rank +=1
        
        if self.generator.rank <= self.generator.depth and self.steps_in_rank==self.args.rank_steps[self.generator.rank]:
            self.steps_in_rank =0
            self.generator.grow()
            self.discriminator.grow()
            
            if self.current_batch_size != self.args.batch_size[self.generator.rank]:
                self.current_batch_size = self.args.batch_size[self.generator.rank]
                self.dataloader = DataLoader(self.dataset,
                                                 batch_size=self.current_batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers
                                                )
                self.iterator = iter(self.dataloader)
            
    
    def save(self, suffix=""):
        postfix = f"iter_{self.n_iter}"
        prefix = self.training_id
        path = f"{self.args.save_dir}/{prefix}_{postfix}{suffix}"

        saves_list = [save for save in os.listdir(self.args.save_dir) if save.split("_")[0] == self.training_id]
        if len(saves_list) >= self.args.num_saves:
            oldest_iter = min([int(save.split("_")[-1]) for save in saves_list])
            save_to_delete = f"{self.training_id}_iter_{oldest_iter}"
            shutil.rmtree(os.path.join(self.args.save_dir, save_to_delete))



        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.discriminator.state_dict(), f"{path}/discriminator")
        torch.save(self.generator.state_dict(), f"{path}/generator")
        torch.save(self.gan_loss.g_optimizer.state_dict(), f"{path}/g_optimizer")
        torch.save(self.gan_loss.d_optimizer.state_dict(), f"{path}/d_optimizer")
        trainer_state_dict = {
            "rank":self.generator.rank,
            "n_iter":self.n_iter,
            "steps_in_rank":self.steps_in_rank,
            "current_batch_size":self.current_batch_size
        }
        with open(f"{path}/trainer", "w", encoding="utf8") as f:
            json.dump(trainer_state_dict, f)

    def load(self, path):
        path = os.path.join(self.args.save_dir, path)
        self.discriminator.load_state_dict(torch.load(os.path.join(path, "discriminator")))
        self.generator.load_state_dict(torch.load(os.path.join(path, "generator")))
        self.gan_loss.d_optimizer.load_state_dict(torch.load(os.path.join(path, "d_optimizer")))
        self.gan_loss.g_optimizer.load_state_dict(torch.load(os.path.join(path, "g_optimizer")))

        with open(os.path.join(path, "trainer"), encoding="utf8") as f:
            trainer_state_dict = json.load(f)
        self.generator.rank = trainer_state_dict["rank"]
        self.discriminator.rank = self.generator.rank
        self.n_iter = trainer_state_dict["n_iter"]
        self.steps_in_rank = trainer_state_dict["steps_in_rank"]
        self.current_batch_size = trainer_state_dict["current_batch_size"]
        