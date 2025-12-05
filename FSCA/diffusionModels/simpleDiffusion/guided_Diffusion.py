from utils.networkHelper import *
from diffusionModels.simpleDiffusion.varianceSchedule import VarianceSchedule

class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model

        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name,
                                                    beta_start=beta_start,
                                                    beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1", condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 传入条件信息给去噪模型
        predicted_noise = self.denoise_model(x_noisy, t, condition)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index, condition=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t, condition) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img,
                                torch.full((b,), i, device=device, dtype=torch.long),
                                i,
                                condition=condition)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, condition=None):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size),
                                  condition=condition)

    def forward(self, mode, **kwargs):
        if mode == "train":
            if "x_start" in kwargs and "t" in kwargs and "condition" in kwargs:
                if "loss_type" in kwargs and "noise" in kwargs:
                    return self.compute_loss(x_start=kwargs["x_start"],
                                             t=kwargs["t"],
                                             noise=kwargs["noise"],
                                             loss_type=kwargs["loss_type"],
                                             condition=kwargs["condition"])
                elif "loss_type" in kwargs:
                    return self.compute_loss(x_start=kwargs["x_start"],
                                             t=kwargs["t"],
                                             loss_type=kwargs["loss_type"],
                                             condition=kwargs["condition"])
                elif "noise" in kwargs:
                    return self.compute_loss(x_start=kwargs["x_start"],
                                             t=kwargs["t"],
                                             noise=kwargs["noise"],
                                             condition=kwargs["condition"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"],
                                             t=kwargs["t"],
                                             condition=kwargs["condition"])
            else:
                raise ValueError(" x_start, t and  condition！")
        elif mode == "generate":
            if "image_size" in kwargs and "batch_size" in kwargs and "channels" in kwargs and "condition" in kwargs:
                return self.sample(image_size=kwargs["image_size"],
                                   batch_size=kwargs["batch_size"],
                                   channels=kwargs["channels"],
                                   condition=kwargs["condition"])
            else:
                raise ValueError(" image_size, batch_size, channels 和 condition ")
        else:
            raise ValueError("mode  'train' or 'generate'")
