import torch

class BaseNoiseScheduler:
    """
    Base class for noise schedulers. 
    """
    def __init__(self):
        
        pass

    def add_noise(self, original, noise, t):
        """
        Forward diffusion: add noise to the original image at timestep t.
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_a_cum_p = self.sqrt_a_cum_p.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_a_cum_p = self.sqrt_one_minus_a_cum_p.to(original.device)[t].reshape(batch_size)

        # Reshape factors to match original tensor dimensions
        for _ in range(len(original_shape) - 1):
            sqrt_a_cum_p = sqrt_a_cum_p.unsqueeze(-1)
            sqrt_one_minus_a_cum_p = sqrt_one_minus_a_cum_p.unsqueeze(-1)

        return (sqrt_a_cum_p * original) + (sqrt_one_minus_a_cum_p * noise)

    def get_sigma(self, eta, t, step_size, device):
        """
        Compute sigma used in DDIM sampling.
        """
        return (eta * self.sqrt_one_minus_a_cum_p.to(device)[t - step_size] / 
                self.sqrt_one_minus_a_cum_p.to(device)[t] * 
                torch.sqrt(1 - self.a_cum_p.to(device)[t] / self.a_cum_p.to(device)[t - step_size]))

    def sample_prev_timestep_ddpm(self, xt, noise_pred, t, noise_intensity=1):
        """
        DDPM backward step. 
        """
        device = xt.device


        mean = xt - ((self.betas.to(device)[t]) * noise_pred) / (self.sqrt_one_minus_a_cum_p.to(device)[t])
        mean = mean / torch.sqrt(self.a_s.to(device)[t])

        if t == 0:
            return mean
        else:
            variance = (1 - self.a_cum_p.to(device)[t - 1]) / (1.0 - self.a_cum_p.to(device)[t])
            variance = variance * self.betas.to(device)[t]
            sigma = noise_intensity * variance ** 0.5
            z = torch.randn(xt.shape).to(device)
            return mean + sigma * z

    def sample_prev_timestep_ddim(self, xt, noise_pred, t, step_size, noise_intensity=0):
        """
        DDIM backward step. Uses eta-based sigma.
        """
        device = xt.device

        x0 = ((xt - (self.sqrt_one_minus_a_cum_p.to(device)[t] * noise_pred)) /
              torch.sqrt(self.a_cum_p.to(device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        sigma = self.get_sigma(noise_intensity, t, step_size, device)
        mean = (self.sqrt_a_cum_p.to(device)[t - step_size] * 
                (xt - self.sqrt_one_minus_a_cum_p.to(device)[t] * noise_pred) /
                self.sqrt_a_cum_p.to(device)[t] +
                torch.sqrt(self.sqrt_one_minus_a_cum_p.to(device)[t - step_size]**2 - sigma**2) * noise_pred)

        if t == 0:
            return mean
        else:
            z = torch.randn(xt.shape).to(device)
            return mean + sigma * z

    def sample_prev_timestep(self, xt, noise_pred, t, step_size=20, sampler_type='ddim'):
        """
        Generic interface for sampling previous timestep.
        """
        if sampler_type == 'ddpm':
            return self.sample_prev_timestep_ddpm(xt, noise_pred, t)
        elif sampler_type == 'ddim':
            return self.sample_prev_timestep_ddim(xt, noise_pred, t, step_size)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")


class LinearNoiseScheduler(BaseNoiseScheduler):
    """
    Linear noise scheduler.
    """
    def __init__(self, num_timesteps, beta_start=torch.tensor(0.0001), beta_end=torch.tensor(0.02)):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.a_s = 1. - self.betas
        self.a_cum_p = torch.cumprod(self.a_s, dim=0)
        self.sqrt_a_cum_p = torch.sqrt(self.a_cum_p)
        self.sqrt_one_minus_a_cum_p = torch.sqrt(1 - self.a_cum_p)


class CosineNoiseScheduler(BaseNoiseScheduler):
    """
    Cosine noise scheduler.
    """
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        a_cum_p_list = []
        for t in range(num_timesteps):
            f = (t/self.num_timesteps + 0.008) / 1.008
            a_cum_p_list.append(torch.cos(torch.tensor(f * torch.pi / 2)) ** 2)
        self.a_cum_p = torch.tensor(a_cum_p_list)

        b_list = [1 - self.a_cum_p[t]/self.a_cum_p[t-1] for t in torch.arange(1, self.num_timesteps)]
        b_list.insert(0, (1/127.5)**2)
        self.betas = torch.tensor([min(i, 0.999) for i in b_list])
        
        self.a_s = 1. - self.betas
        self.sqrt_a_cum_p = torch.sqrt(self.a_cum_p)
        self.sqrt_one_minus_a_cum_p = torch.sqrt(1 - self.a_cum_p)


class StableDiffusionNoiseScheduler(BaseNoiseScheduler):
    """
    Linear noise scheduler.
    """
    def __init__(self, num_timesteps, beta_start=torch.sqrt(torch.tensor(0.00085)), beta_end=torch.sqrt(torch.tensor(0.012))):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.square(torch.linspace(beta_start, beta_end, num_timesteps))
        self.a_s = 1. - self.betas
        self.a_cum_p = torch.cumprod(self.a_s, dim=0)
        self.sqrt_a_cum_p = torch.sqrt(self.a_cum_p)
        self.sqrt_one_minus_a_cum_p = torch.sqrt(1 - self.a_cum_p)


