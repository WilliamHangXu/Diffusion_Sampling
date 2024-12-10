import torch
import torchvision
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.unet import UNet
from diffusion.schedulers import LinearNoiseScheduler, CosineNoiseScheduler, StableDiffusionNoiseScheduler
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_config):
   
    
    num_timesteps = 1000
    scheduler_type = train_config['noise_scheduler']
    num_epochs = train_config['num_epochs']
    job_name = '{}_{}'.format(scheduler_type, num_epochs)

    
    match scheduler_type:
        case 'linear':
            scheduler = LinearNoiseScheduler(num_timesteps)
        case 'cosine':
            scheduler = CosineNoiseScheduler(num_timesteps)
        case 'sd':
            scheduler = StableDiffusionNoiseScheduler(num_timesteps)
        case _:
            scheduler = LinearNoiseScheduler(num_timesteps)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    cifar10_loader = DataLoader(cifar10, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    


    model = UNet().to(device)
    model.train()
    
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    epoch_loss = []
    losses = []

    for epoch_idx in range(num_epochs):
        
        average_train_loss = 0
        loop_train = tqdm(enumerate(cifar10_loader, 1), total=len(cifar10_loader), desc="Train", position=0, leave=True)
        for index, im in loop_train:
            optimizer.zero_grad()
            im = im[0]
            im = im.float().to(device)
            
            noise = torch.randn_like(im).to(device)
            
            t = torch.randint(0, num_timesteps, (im.shape[0],)).to(device)
            
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            average_train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            loop_train.set_description(f"Train - iteration : {epoch_idx+1}")
            loop_train.set_postfix(
                avg_train_loss="{:.4f}".format(average_train_loss / index),
                refresh=True,
            )
        epoch_loss.append(average_train_loss / len(cifar10_loader))
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        if (epoch_idx + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join('weights', '{}_{}.pt'.format(job_name, epoch_idx + 1)))
    plt.plot(losses)
    plt.title('Iteration Loss')
    plt.savefig('train_stats/{}_iter_loss.png'.format(job_name))
    plt.clf()
    plt.plot(epoch_loss)
    plt.title('Epoch Loss')
    plt.savefig('train_stats/{}_epoch_loss.png'.format(job_name))
    plt.clf()
    print('Done Training ...')
    np.savetxt("train_stats/{}_iter_loss.txt".format(job_name), losses, fmt="%.6f")
    np.savetxt("train_stats/{}_epoch_loss.txt".format(job_name), epoch_loss, fmt="%.6f")
    

if __name__ == '__main__':
    
    train_config = {
        'noise_scheduler': 'cosine',
        'batch_size': 64,
        'num_epochs': 200,
        'num_samples' : 100,
        'num_grid_rows' : 10,
        'lr': 0.0001
    }
    train(train_config)
