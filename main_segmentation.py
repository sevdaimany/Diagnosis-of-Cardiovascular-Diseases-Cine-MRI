import torch
import numpy as np
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader,TensorDataset
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.args_utils import parse_args
from utils.data_utils import get_acdc,convert_masks
from utils.model import FCT
from utils.dataset_utils import ACDCTrainDataset
import os

args = parse_args()

def get_lr_scheduler(args,optimizer):
    if args.lr_scheduler == 'none':
        return None
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            verbose=True,
            threshold=1e-6,
            patience=5,
            min_lr=args.min_lr)
        return scheduler
    if args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=500
        )
        return scheduler

@torch.no_grad()
def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    torch.save(model, model_path)
    return model_path

def main():
    # model instatation
    model = FCT(args)
    model.apply(init_weights)

    # get data
    acdc_data, _, _ = get_acdc('ACDC/training', input_size=(args.img_size,args.img_size,1))

    # split data into train and val
    X_train, X_val, y_train, y_val = train_test_split(acdc_data[0], acdc_data[1], test_size=0.2, random_state=42)
    train_dataset = ACDCTrainDataset(X_train, y_train,args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers)

    # validation
    y_val = convert_masks(y_val)
    X_val = np.transpose(X_val, (0, 3, 1, 2)) # for the channels
    y_val = np.transpose(y_val, (0, 3, 1, 2)) # for the channels
    X_val = torch.Tensor(X_val) # convert to tensors
    y_val = torch.Tensor(y_val) # convert to tensors
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.workers)

    # resume
    if args.resume:
        if args.new_param:
            model = FCT.load_from_checkpoint('lightning_logs/version_2/checkpoints/epoch=74-step=4500.ckpt',args=args)
        else:
            # load weights,old hyper parameter and optimizer state 
            model = FCT.load_from_checkpoint('this is path')
    
    precision = '16-mixed' if args.amp else 32
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    save_every_n_epoch = ModelCheckpoint(every_n_epochs=10)


    trainer = L.Trainer(precision=precision,max_epochs=args.max_epoch,callbacks=[lr_monitor, save_every_n_epoch])
    trainer.fit(model=model,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
    # save_model(model, "models/fct.model")

if __name__ == '__main__':
    main()