import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

from transformer import PointDiffusionTransformer


class Diffusion(nn.Module):
    
    def __init__(self, device, timesteps=1000, max_points=1400):
        super().__init__()
        
        self.timesteps = timesteps
        self.max_points = max_points
        self.device = device
        
        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        
        self.network = PointDiffusionTransformer(device, n_ctx=self.max_points)
    
    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas
    
    def _prepare_data(self, points, GT_lines=None):
        bs = len(points)
        device = points[0].device
        
        if GT_lines is not None:
            labels = []
            for i in range(bs):
                line_2pt = GT_lines[i].view(-1, 2)
                points[i] = torch.cat((points[i], line_2pt), dim=0)
                
                GT_num = GT_lines[i].shape[0]
                point_num = points[i].shape[0]
                mask = torch.zeros(point_num).to(device)
                
                for j in range(GT_num):
                    mask[point_num - j * 2 - 1] = (j + 1) / GT_num
                    mask[point_num - j * 2 - 2] = (j + 1) / GT_num
                
                shuffle = torch.randperm(point_num)
                points[i] = points[i][shuffle]
                mask = mask[shuffle]
                labels.append(mask)
            
            labels = pad_sequence(labels, batch_first=True, padding_value=0)
        else:
            labels = [torch.zeros(p.shape[0]).to(device) for p in points]
            labels = pad_sequence(labels, batch_first=True, padding_value=0)
        
        points = pad_sequence(points, batch_first=True, padding_value=0)
        padding_needed = self.max_points - points.shape[1]
        
        points = F.pad(points, (0, 0, 0, padding_needed), "constant", 0)
        labels = F.pad(labels, (0, padding_needed), "constant", 0)
        
        return points.permute(0, 2, 1), labels.unsqueeze(1)
    
    def forward(self, points, GT_lines=None, training=True, **kwargs):
        device = points[0].device
        bs = len(points)
        
        points_tensor, labels = self._prepare_data(points, GT_lines)
        
        if training:
            return self._forward_train(points_tensor, labels, bs, device)
        else:
            return self._forward_inference(points_tensor, labels, bs, device, GT_lines, points)
    
    def _forward_train(self, points_tensor, labels, bs, device):
        t = torch.randint(0, self.timesteps, (bs,)).to(device)
        
        points_tensor = points_tensor * 2. - 1.
        labels = labels * 2. - 1.
        
        noise = torch.randn_like(labels).to(device)
        x_t = self._forward_diffusion(labels, t, noise)
        
        pred_noise = self.network(points_tensor, x_t, t)
        
        return pred_noise, noise
    
    def _forward_inference(self, points_tensor, labels, bs, device, GT_lines, points):
        point_nums = [p.shape[0] for p in points]
        GT_nums = [gt.shape[0] for gt in GT_lines]
        
        points_tensor = points_tensor * 2. - 1.
        
        x_t = torch.randn_like(labels).to(device)
        
        for i in range(self.timesteps - 1, -1, -1):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i] * bs).to(device)
            x_t = self._reverse_diffusion_with_clip(points_tensor, x_t, t, noise)
        
        return self._parse_results(x_t, point_nums, GT_nums, points_tensor)
    
    def _forward_diffusion(self, x_0, t, noise):
        return (
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1) * x_0 +
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1) * noise
        )
    
    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, points, x_t, t, noise):
        pred = self.network(points, x_t, t)
        
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1)
        
        x_0_pred = (
            torch.sqrt(1. / alpha_t_cumprod) * x_t -
            torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        )
        x_0_pred.clamp_(-1., 1.)
        
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1)
            mean = (
                (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred +
                ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
            )
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred
            std = 0.0
        
        return mean + std * noise
    
    def _parse_results(self, x_t, point_nums, GT_nums, points_tensor):
        pred_lines = []
        
        for i in range(len(point_nums)):
            x_t_i = x_t[i, :, :point_nums[i]]
            x_t_i = torch.softmax(x_t_i, dim=1)
            
            _, indices = torch.sort(x_t_i, dim=1, descending=True)
            selected_points = indices[:, :GT_nums[i] * 2]
            
            pred_lines_i = []
            for j in range(GT_nums[i]):
                p1_idx = selected_points[0, j * 2]
                p2_idx = selected_points[0, j * 2 + 1]
                
                line = torch.cat([
                    points_tensor[i, :, p1_idx],
                    points_tensor[i, :, p2_idx]
                ]).unsqueeze(0)
                pred_lines_i.append(line)
            
            if pred_lines_i:
                pred_lines.append(torch.cat(pred_lines_i, dim=0))
            else:
                pred_lines.append(torch.empty(0, 4).to(x_t.device))
        
        return pred_lines