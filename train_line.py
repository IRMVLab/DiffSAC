import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
import time
import wandb
from tqdm import tqdm

from diffusion import Diffusion
from dataset import Dataset_line, line_dataset_collate
from utils import eval_line, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="DiffSAC Line Detection Training")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--timesteps', type=int, default=1000, help='DDPM timesteps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--train_data', type=str, default='../Dataset_line/train/06', help='Training data path')
    parser.add_argument('--val_data', type=str, default='../Dataset_line/val/06', help='Validation data path')
    parser.add_argument('--wandb_mode', type=str, default='dryrun', choices=['online', 'offline', 'dryrun'])
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    
    with tqdm(total=len(dataloader), desc=f'Train Epoch {epoch}') as pbar:
        for step, (points, lines, _, _) in enumerate(dataloader):
            points = [point.to(device) for point in points]
            GT_lines = [line[:, :4].to(device) for line in lines]
            
            pred_label, gt_label = model(points, GT_lines=GT_lines, training=True)
            loss = loss_fn(pred_label, gt_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            pbar.update(1)
            
            if step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "step": epoch * len(dataloader) + step})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_pred_lines, total_gt_lines = [], []
    
    with tqdm(total=len(dataloader), desc=f'Eval Epoch {epoch}') as pbar:
        with torch.no_grad():
            for points, lines, _, _ in dataloader:
                points = [point.to(device) for point in points]
                GT_lines = [line[:, :4].to(device) for line in lines]
                
                pred_lines = model(points, GT_lines=GT_lines, training=False)
                total_pred_lines.extend(pred_lines)
                total_gt_lines.extend(GT_lines)
                
                pbar.update(1)
    
    auc = eval_line(total_gt_lines, total_pred_lines)
    wandb.log({"val_auc": auc, "epoch": epoch})
    print(f"Epoch {epoch} - AUC: {auc:.4f}")
    
    return auc


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = Dataset_line(args.train_data, data_num='all')
    val_dataset = Dataset_line(args.val_data, data_num='all')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=line_dataset_collate, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=line_dataset_collate, pin_memory=True
    )
    
    model = Diffusion(device=device, timesteps=args.timesteps).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    current_time = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = f'checkpoints/difflinesac-{current_time}/'
    
    wandb.init(
        project='difflinesac',
        name=f'difflinesac-{current_time}',
        mode=args.wandb_mode,
        config=vars(args)
    )
    
    start_epoch = 0
    best_auc = 0
    if args.checkpoint and torch.load(args.checkpoint, map_location='cpu'):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_auc = checkpoint.get('best_auc', 0)
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        if epoch % args.eval_interval == 0:
            auc = evaluate(model, val_loader, device, epoch)
            
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'args': vars(args)
                }, f'{log_dir}/best_model.pth')
            
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'args': vars(args)
                }, f'{log_dir}/checkpoint_epoch_{epoch}.pth')
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()