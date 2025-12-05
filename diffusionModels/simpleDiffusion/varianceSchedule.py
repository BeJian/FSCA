from utils.networkHelper import *

def cosine_beta_schedule(timesteps, s=0.008, **kwargs):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

# def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
#     betas = torch.linspace(-3, 3, timesteps)
#     return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def sigmoid_beta_schedule(timesteps, beta_start = -3, beta_end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(beta_start / tau).sigmoid()
    v_end = torch.tensor(beta_end / tau).sigmoid()
    alphas_cumprod = (-((t * (beta_end - beta_start) + beta_start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class VarianceSchedule(nn.Module):
    def __init__(self, schedule_name="linear_beta_schedule", beta_start=None, beta_end=None):
        super(VarianceSchedule, self).__init__()
        self.schedule_name = schedule_name
        beta_schedule_dict = {
            'linear_beta_schedule': linear_beta_schedule,
            'cosine_beta_schedule': cosine_beta_schedule,
            'quadratic_beta_schedule': quadratic_beta_schedule,
            'sigmoid_beta_schedule': sigmoid_beta_schedule
        }
        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')
        if beta_end and beta_start is None and schedule_name != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        if self.schedule_name == "cosine_beta_schedule":
            return self.selected_schedule(timesteps=timesteps)
        else:
            return self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
