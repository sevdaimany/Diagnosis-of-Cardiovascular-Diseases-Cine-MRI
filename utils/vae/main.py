from torch.utils.data import DataLoader
# from dataset import get_loader
from utils.vae.dataset import get_loader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from sklearn.metrics import  accuracy_score

def evaluate(vae: nn.Module, dataloader:DataLoader, writer:SummaryWriter, start_epoch:int):
    vae.eval()
    total_loss = 0
    total_cce = 0
    total_acc = 0
    with torch.no_grad():
        for batch_idx, (batch_data, ground_truth) in enumerate(dataloader):
            ground_truth = ground_truth.cuda()
            batch_data = batch_data.cuda()
            recon_batch, mu, log_var = vae(batch_data)
            pred = vae.forward_mlp(mu)
            loss, cce = loss_function(recon_batch, batch_data, mu, log_var, pred, ground_truth)
            total_loss += loss.item()
            total_cce += cce.item()
            pred = np.argmax(pred.cpu().detach().numpy(), axis=-1).tolist()
            acc = accuracy_score(ground_truth.cpu().detach().numpy().tolist(), pred)  
            total_acc += acc
    avg_loss = total_loss / len(dataloader)
    avg_cce = total_cce / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    
    writer.add_scalar("val/total_loss_epoch", avg_loss, start_epoch)
    writer.add_scalar("val/cce_loss_epoch", avg_cce, start_epoch)
    writer.add_scalar("val/accuracy_epoch", avg_acc, start_epoch)
    
    print('Validation Loss: {:.4f}'.format(avg_loss))

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 8) * (input_dim // 8), latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * (input_dim // 8) * (input_dim // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (64, input_dim // 8, input_dim // 8)),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 5)
        )
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(mu.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var
    
    def forward_mlp(self, mu):
        edmu, esmu = torch.chunk(mu, chunks=2, dim=0)
        pred = torch.cat([edmu, esmu], dim=1)
        return self.mlp(pred)
    
    def bottle_neck(self, x):
        mu, log_var = self.encode(x)
        return mu
    

criteria = nn.CrossEntropyLoss()
def loss_function(recon_x, x, mu, log_var, pred, y):
    BCE = F.mse_loss(recon_x, x,  reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    CCE = criteria(pred, y)
    loss_vae = BCE + KLD
    return loss_vae +  1000 *CCE , 1000 * CCE



if __name__ == "__main__":
    now = datetime.now()
    run_name = f'{now.strftime("%Y_%m_%d_%H_%M_%S")}'
    writer = SummaryWriter(os.path.join("model_zoo2/tensorboard", run_name))
        
    trainloader = get_loader(dir="data/train", csv="train_patients_info.csv", label_encoder_path=None, batch_size=8)
    testloader = get_loader(dir="data/test", csv="test_patients_info.csv", label_encoder_path="", batch_size=8)

    input_dim = 224
    latent_dim = 16
    resume = None
    start_epoch = 0
    end_epoch = 0
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    if resume is not None:
        start_epoch = int(resume.split("-")[0])
        end_epoch = int(resume.split("-")[1])
        vae.load_state_dict(torch.load(f"model_zoo2/Epoch[{resume}].pth"))
    vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)


    num_epochs = 400
    vae.train()
    for epoch in range(start_epoch + 1, num_epochs+end_epoch+1):
        total_loss = 0
        total_cce = 0
        total_acc = 0
        for batch_idx, (batch_data, ground_truth) in tqdm(enumerate(trainloader)):
            optimizer.zero_grad()
            b_d = batch_data.cuda()
            ground_truth = ground_truth.cuda()
            recon_batch, mu, log_var = vae(b_d)
            pred = vae.forward_mlp(mu)
            
            loss, cce = loss_function(recon_batch, b_d, mu, log_var, pred, ground_truth)
            writer.add_scalar("train/total_loss_step", loss.item(), batch_idx + epoch * len(trainloader))
            writer.add_scalar("train/cce_loss_step", cce.item(), batch_idx + epoch * len(trainloader))
            
            loss.backward()
            total_loss += loss.item()
            total_cce += cce.item()
            
            pred = np.argmax(pred.cpu().detach().numpy(), axis=-1).tolist()
            acc = accuracy_score(ground_truth.cpu().detach().numpy().tolist(), pred)  
            writer.add_scalar("train/accuracy_step", acc, batch_idx + epoch * len(trainloader))
            
            total_acc += acc
            optimizer.step()
        
        writer.add_scalar("train/total_loss_epoch", total_loss/len(trainloader), epoch)
        writer.add_scalar("train/cce_loss_epoch", total_cce/len(trainloader), epoch)
        writer.add_scalar("train/accuracy_epoch", total_acc/len(trainloader), epoch)

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / len(trainloader)))
        torch.save(vae.state_dict(), f"model_zoo2/Epoch[{epoch+1}-{num_epochs+end_epoch+1}].pth")
        evaluate(vae, testloader, writer, epoch)
        batch_data, ground_truth = next(iter(testloader))
        input_batch = batch_data
        gt_batch = input_batch.numpy()
        recon_batch, _, _ = vae(input_batch.cuda())
        recon_batch = recon_batch.cpu().detach().numpy()

        # Visualize the results for a random sample
        idx = np.random.randint(0, input_batch.size(0))
        # gt = np.argmax(gt_batch[idx].transpose(1, 2, 0), axis=-1)
        # rc = np.argmax(recon_batch[idx].transpose(1, 2, 0), axis=-1)
        gt = (gt_batch[idx,0] * 255).astype(np.uint8)
        rc = (recon_batch[idx,0] * 255).astype(np.uint8)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Ground Truth')
        plt.imshow(gt, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title('Reconstructed')
        plt.imshow(rc, cmap="gray")
        plt.savefig(f"model_zoo2/res{epoch+1}.png")
        plt.close('all')