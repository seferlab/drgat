
import torch, math
import torch.nn as nn

def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class MLPBackbone(nn.Module):
    def __init__(self, latent_dim, cond_dim=2, time_dim=128, hidden=256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_emb = nn.Embedding(cond_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(latent_dim+hidden+hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.time_dim = time_dim
    def forward(self, x, t, c):
        te = timestep_embedding(t, self.time_dim)
        te = self.time_mlp(te)
        ce = self.cond_emb(c)
        h = torch.cat([x, te, ce], dim=-1)
        return self.net(h)

class DDIMLatent(nn.Module):
    def __init__(self, latent_dim, cond_dim=2, timesteps=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = MLPBackbone(latent_dim, cond_dim=cond_dim)
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sqrt_ac = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_om = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
        return sqrt_ac * x0 + sqrt_om * noise

    def training_losses(self, x0, c):
        b = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_hat = self.model(xt, t, c)
        return ((noise - eps_hat) ** 2).mean()

    @torch.no_grad()
    def ddim_sample(self, shape, c, eta=0.0):
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = self.model(x, tt, c)
            a_t = self.alphas[t]
            ac_t = self.alphas_cumprod[t]
            # x0 estimate
            x0 = (x - torch.sqrt(1 - ac_t) * eps) / torch.sqrt(ac_t + 1e-8)
            if t == 0:
                x = x0
            else:
                ac_tm1 = self.alphas_cumprod[t-1]
                sigma_t = eta * torch.sqrt((1 - ac_tm1)/(1 - ac_t) * (1 - a_t))
                dir_term = torch.sqrt(1 - ac_tm1 - sigma_t**2) * eps
                x = torch.sqrt(ac_tm1) * x0 + dir_term + sigma_t * torch.randn_like(x)
        return x
