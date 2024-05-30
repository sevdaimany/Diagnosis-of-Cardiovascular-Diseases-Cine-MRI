from torch.utils.data import Dataset, DataLoader
import torch
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# trainset = MyDataset(dir="data/train", csv="train_patients_info.csv", label_encoder_path=None)
# testset = MyDataset(dir="data/test", csv="test_patients_info.csv",label_encoder_path="")
class MyDataset(Dataset):
    def __init__(self, dir, csv, label_encoder_path=None) -> None:
        super().__init__()
        self.files = sorted(glob.glob(f"{dir}/*.nii.gz"))
        self.files = [(self.files[i], self.files[i+1]) for i in range(0, len(self.files), 2)]
        self.csv = pd.read_csv(csv)[["patient_id","Group"]]
        self.y = []
        self.patinets = {}
        for i in range(len(self.csv)):
            self.y.append(self.csv.iloc[i]["Group"])
            self.patinets[str(int(self.csv.iloc[i]["patient_id"]))] = i
        
        if label_encoder_path is None:
            labelencoder = LabelEncoder()
            self.y = labelencoder.fit_transform(self.y)
            joblib.dump(labelencoder, "transformers/labelencoder.joblib")
        else:
            labelencoder = joblib.load("transformers/labelencoder.joblib")
            self.y = labelencoder.transform(self.y)

    
    def load_path(self, path):
        x =  nib.load(path).get_fdata()[None, ...]
        return x
    
    def get_item(self, path):
        return self.preprocess(torch.nn.functional.interpolate(torch.from_numpy(self.load_path(path)).permute(3, 0, 1, 2), (224, 224)).to(int)).to(torch.float32)
    
    def __getitem__(self, index):
        pid = str(int(self.files[index][0].split("patient")[-1][:3]))
        ed = self.get_item(self.files[index][0])
        es = self.get_item(self.files[index][1])
        return ed, es, [self.y[self.patinets[pid]]] * ed.shape[0]
    
    def __len__(self):
        return len(self.files)
    
    def preprocess(self, item):
        return item / 3


def collate_fn(batch):
    ed = []
    es = []
    y = []
    for b in batch:
        ed.append(b[0])
        es.append(b[1])
        y.extend(b[2])
    return torch.cat((torch.cat(ed, dim=0).float(), torch.cat(es, dim=0).float()), dim=0), torch.from_numpy(np.array(y)).to(torch.int64)

def get_loader(dir, csv, label_encoder_path, batch_size):
    dataset = MyDataset(dir=dir, csv=csv, label_encoder_path=label_encoder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader