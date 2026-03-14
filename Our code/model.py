import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """Positional encdoing for scalar time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-torch.arange(half).float() * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        args = t[:, None] * self.freqs[None]          # (B, half)
        emb = torch.cat([args.sin(), args.cos()], -1)  # (B, dim)
        return self.proj(emb)                          # (B, dim)


class ResBlock(nn.Module):
    """Residual block conditioned on time embedding."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, time_dim)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.down(x), x          # return (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(out_ch + skip_ch, time_dim)
        self.proj = nn.Conv2d(out_ch + skip_ch, out_ch, 1)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return self.proj(x)

""" U-net velocity network """

class VelocityUNet(nn.Module): 
    """ 
    should predict velocity v(x_t, t) for Rectified Flow on 28x28 grayscale images.
    Architecture: a shallow U-net with three resolution levels.
    Channels: 32 -> 64 -> 128, time conditioning occurs at every residual block
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # encoder
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim)

        # bottleneck
        self.mid1 = ResBlock(base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, time_dim)

        # decoder
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, time_dim)
        self.up2 = UpBlock(base_ch * 2, base_ch, base_ch, time_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments: 
            x: noisy image (B, 1, 28, 28)
            t: time in [0, 1]  (B,)
        Returns: 
            velocity field (B, 1, 28, 28)
        """
        # (B, time_dim)
        t_emb = self.time_emb(t)

        # encoder
        h = self.in_conv(x)                  # (B, 32, 28, 28)
        h, skip1 = self.down1(h, t_emb)     # (B, 64, 14, 14)
        h, skip2 = self.down2(h, t_emb)     # (B, 128, 7, 7)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # decoder
        h = self.up1(h, skip2, t_emb)       # (B, 64, 14, 14)
        h = self.up2(h, skip1, t_emb)       # (B, 32, 28, 28)

        return self.out_conv(F.silu(self.out_norm(h)))

    # Rectified Flow Wrapper bit

class RectifiedFlow(nn.Module):
    """
    Wraps the velocityunet with rectified flow training and sampling logic. 

    Training goal: 
        L = E_{x~data, z~N(0,I), t~U[0,1]} [ ||v_theta(x_t, t) - (x - z) ||^2]
        where x_t = (1-t)*z + t*x

    Sampling (Euler with a given schedule {t_k}):
        x_{t_{k+1}} = x_{t_k} + (t_{k+1} - t_k) * v_theta(x_{t_k}, t_k)
    """

    def __init__(self, net: VelocityUNet): 
        super().__init__()
        self.net = net

    def get_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the rectified flow training loss for a batch of clean images x0
        """
        B = x0.size(0)
        device = x0.device

        # source noise
        z = torch.randn_like(x0)

        #t ~ U[0,1]
        t = torch.rand(B, device=device)
        t4d = t.view(B, 1, 1, 1)

        # linear interpolation
        x_t = (1 - t4d) * z + t4d * x0
        # true velocity
        target = x0 -z

        pred = self.net(x_t, t)
        return F.mse_loss(pred, target)
    
    @torch.no_grad()
    def sample(
        self, 
        n: int, 
        schedule: torch.Tensor,
        device: torch.device, 
    ) -> torch.Tensor: 
        
        """
        Generate n amount of samples using the Euler integration along a time schedule. 

        Arguments: 
            n: Number of samples
            schedule: 1-D tensor of increasing time values in [0,1]
                (example: torch.linspace(0, 1, steps+1))
            device: target device
        
        Reurtns: 
            generated images (n, 1, 28, 28) in [-1, 1]
        """
        x = torch.randn(n, 1, 28, 28, device=device)
        schedule = schedule.to(device)

        for i in range(len(schedule) - 1):
            t_cur = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_cur

            t_batch = t_cur.expand(n)
            v = self.net(x, t_batch)
            x = x + dt * v
        return x.clamp(-1, 1)
    
    @torch.no_grad()
    def compute_flow_matching_error(
        self, 
        x0: torch.Tensor,
        t: float, 
        n_samples: int = 512,
    ) -> float:
        
        """
        Estimate teh score matching analogue for RF at the time t:
            use: FME(t) = E_{x,z} [ ||v_theta(x_t, t) - (x - z)||^2 ]

        This is the ϵ(t) used in EB for the adaptive schedule algorithms.
        """

        device = x0.device
        indices = torch.randint(0, x0.size(0), (n_samples,))
        x_sub = x0[indices]
        z = torch.randn_like(x_sub)
        t_tensor = torch.full((n_samples,), t, device=device)
        t4d = t_tensor.view(n_samples, 1, 1, 1)

        x_t = (1 - t4d) * z + t4d * x_sub
        target = x_sub - z
        pred = self.net(x_t, t_tensor)

        return F.mse_loss(pred, target).item()