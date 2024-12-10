import torch
import torchvision
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from model.unet import UNet
from diffusion.schedulers import LinearNoiseScheduler, CosineNoiseScheduler, StableDiffusionNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(samp_idx, model, sample_config):
    
    sampler_type = sample_config['sampler']
    scheduler_type = sample_config['noise_scheduler']
    weight = sample_config['weight'][0:-3]
    num_timesteps = 1000
    dir_name = '{}_{}'.format(sampler_type, weight)
    step_size = 1 if sampler_type == 'ddpm' else 20
    xt = torch.randn((100, 3, 32, 32)).to(device)
    
    match scheduler_type:
        case 'linear':
            scheduler = LinearNoiseScheduler(num_timesteps)
        case 'cosine':
            scheduler = CosineNoiseScheduler(num_timesteps)
        case 'sd':
            scheduler = StableDiffusionNoiseScheduler(num_timesteps)
        case _:
            scheduler = LinearNoiseScheduler(num_timesteps)
    
    
    for i in tqdm(reversed(range(step_size, num_timesteps, step_size))):

        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        xt = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device), step_size, sampler_type)

    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, 10)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join('results', dir_name)):
        os.mkdir(os.path.join('results', dir_name))
    img.save(os.path.join('results', dir_name, 'x0_{}.png'.format(samp_idx)))
    img.close()


def sample_batch(sample_config):
    unet = UNet()
    model = unet.to(device)
    model.load_state_dict(torch.load(os.path.join('weights', sample_config['weight']), map_location=device))
    model.eval()
    
    with torch.no_grad():
        for i in range(sample_config['samp_time']):
            print("Sample batch {}".format(i))
            sample(i, model, sample_config)


if __name__ == '__main__':
 
    sample_config = {
        'weight': 'linear_2040.pt',
        'sampler': 'ddim',
        'noise_scheduler': 'linear',
        'samp_time': 1
    }

    sample_batch(sample_config)
