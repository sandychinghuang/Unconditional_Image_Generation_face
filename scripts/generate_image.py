import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
        )
        self.act = nn.SiLU()
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.act(self.block(x) + self.residual(x))

def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.norm(x)
        x_ = x_.reshape(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.attn(x_, x_, x_)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + attn_out

class SkipAttention(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.norm_q = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, decoder_feat, encoder_feat):
        B, C, H, W = decoder_feat.shape
        q = self.norm_q(decoder_feat).reshape(B, C, H * W).permute(0, 2, 1)
        kv = self.norm_kv(encoder_feat).reshape(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.attn(q, kv, kv)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return decoder_feat + attn_out

class AttentionUNet(nn.Module):
    def __init__(self, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.time_proj1 = nn.Linear(time_emb_dim, 64)
        self.time_proj2 = nn.Linear(time_emb_dim, 128)
        self.time_proj3 = nn.Linear(time_emb_dim, 256)
        self.time_proj4 = nn.Linear(time_emb_dim, 512)
        self.time_proj_up3 = nn.Linear(time_emb_dim, 256)
        self.time_proj_up2 = nn.Linear(time_emb_dim, 128)
        self.time_proj_up1 = nn.Linear(time_emb_dim, 64)

        # Encoder
        self.down1 = ResBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(256, 512)
        self.attn = SelfAttention(512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.skip_attn3 = SkipAttention(256)
        self.dec3 = ResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.skip_attn2 = SkipAttention(128)
        self.dec2 = ResBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResBlock(128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)

        x1 = self.down1(x) + self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x2 = self.down2(self.pool1(x1)) + self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x3 = self.down3(self.pool2(x2)) + self.time_proj3(t_emb).unsqueeze(-1).unsqueeze(-1)

        x_bottleneck = self.bottleneck(self.pool3(x3)) + self.time_proj4(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_bottleneck = self.attn(x_bottleneck)

        x = self.up3(x_bottleneck)
        x3 = self.skip_attn3(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x) + self.time_proj_up3(t_emb).unsqueeze(-1).unsqueeze(-1)

        x = self.up2(x)
        x2 = self.skip_attn2(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x) + self.time_proj_up2(t_emb).unsqueeze(-1).unsqueeze(-1)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x) + self.time_proj_up1(t_emb).unsqueeze(-1).unsqueeze(-1)

        return self.final(x)

class Diffusion_DDIM:
    def __init__(self, T=1000, device="cuda"):
        self.device = device
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


    def sample(self, model, img_size, batch_size, num_steps=50):
        model.eval()
        step_size = self.T // num_steps
        x = torch.randn((batch_size, 3, img_size, img_size)).to(self.device)

        for t in reversed(range(0, self.T, step_size)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = model(x, t_batch)

            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t - step_size] if t - step_size >= 0 else torch.tensor(1.0).to(self.device)

            x0_pred = (x - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            x0_pred = torch.clamp(x0_pred, -1., 1.)

            sigma = 0  # DDIM: deterministic (no noise)
            dir_xt = ((1 - alpha_bar_prev).sqrt()) * predicted_noise
            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)

        return x


# 直接生成10000張
device = "cuda" if torch.cuda.is_available() else "cpu"

model_gen_path = r"model4_att_Res/ema_unet_epoch200.pth"  #需更改為模型路徑
model_gen = AttentionUNet().to(device)
model_gen.load_state_dict(torch.load(model_gen_path))

model_gen.eval()
diffusion = Diffusion_DDIM(device=device)

os.makedirs("generated_images", exist_ok=True)

# 依序載入每個模型並生成一張圖片
for i in range(10000):
    with torch.no_grad():
        sample = diffusion.sample(model_gen, img_size=64, batch_size=1, num_steps=50)[0]
        sample = sample * 0.5 + 0.5  # 去 normalize
    save_image(sample, f"generated_images/img_{i}.png")
        