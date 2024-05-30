import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from torch.utils.data import Dataset
from .data_utils import convert_mask_single

class ClassificationDataset(Dataset):
  def __init__(self, x, manual_features, y):
    super().__init__()
    self.x = x
    self.manual_features = manual_features
    self.y = y
    self.img_size = 224
    self.img_size2 = 128
    
    

  def __len__(self):
        return self.x.shape[0]
    
  def __getitem__(self, index):
      return torch.tensor(self.x[index]).float(),torch.tensor(self.manual_features[index]).float(), torch.tensor(self.y[index])


class ACDCTrainDataset(Dataset):
    def __init__(self,x,y,args) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=20,translate=(0.2,0.2))
        ])
        self.img_size = args.img_size
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        seed = np.random.randint(2147483647)

        x = PIL.Image.fromarray(self.x[index].reshape(self.img_size, self.img_size))
        y = PIL.Image.fromarray(self.y[index].reshape(self.img_size, self.img_size))

        torch.manual_seed(seed)
        tar_x = np.array(self.transform(x))

        torch.manual_seed(seed)
        tar_y = np.array(self.transform(y))
        tar_y = convert_mask_single(tar_y)
        tar_x = tar_x.reshape(1,self.img_size,self.img_size)
        torch.manual_seed(0)
        return torch.tensor(tar_x).float(),torch.tensor(tar_y).float()