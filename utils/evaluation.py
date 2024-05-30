import os
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import pandas as pd
from utils.data_utils import get_acdc,convert_masks

def compute_dice(pred_y, y):
    """
    Computes the Dice coefficient for each class in the ACDC dataset.
    Assumes binary masks with shape (num_masks, num_classes, height, width).
    """
    epsilon = 1e-6
    num_masks = pred_y.shape[0]
    num_classes = pred_y.shape[1]
    device = torch.device("cuda")
    dice_scores = torch.zeros((num_classes,), device=device)

    for c in range(num_classes):
        intersection = torch.sum(pred_y[:, c] * y[:, c])
        sum_masks = torch.sum(pred_y[:, c]) + torch.sum(y[:, c])
        dice_scores[c] = (2. * intersection + epsilon) / (sum_masks + epsilon)

    return dice_scores   
    
def calculate_dice_foreach_patient(model, dataloader, patient_id, scores):
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)
    slice_id = 1
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        y_pred = torch.argmax(outputs[2], axis=1)

        # compute dice
        # convert to 4 channels to compare with gt, since gt has 4 channels
        y_pred_onehot = F.one_hot(y_pred, 4).permute(0, 3, 1, 2)
        dice = compute_dice(y_pred_onehot, targets)
        dice_lv = dice[3].item()
        dice_rv = dice[1].item()
        dice_myo = dice[2].item()
        # skip background for mean
        dice_avg = dice[1:].mean().item()

        scores.append({'patient_id': patient_id,
                'slice_id': slice_id,
                'dice_avg': dice_avg,
                'dice_lv': dice_lv,
                'dice_rv': dice_rv,
                'dice_myo': dice_myo})
        
        slice_id += 1
 
def evaluate(path, model):
    scores = []
    for root, directories, files in os.walk(path):
        if "patient" in root:
            patient_id = root[-3:]
            print(root)
            acdc_data, _, _ = get_acdc(root, input_size=(224, 224, 1))
            acdc_data[1] = convert_masks(acdc_data[1])
            acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
            acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
            acdc_data[0] = torch.Tensor(acdc_data[0]) # convert to tensors
            acdc_data[1] = torch.Tensor(acdc_data[1]) # convert to tensors
            acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
            data_loader = DataLoader(acdc_data, batch_size=1, num_workers=2)
            calculate_dice_foreach_patient(model, data_loader, patient_id, scores)
    scores = pd.DataFrame(scores)
    return scores
                    
                    
    
    
if __name__ == "__main__":
    path = 'ACDC/training/'
    
    export_path = 'results/fct_scores_train_2.csv'
    # model = torch.load('models/fct.model')
    model = torch.load('check/model/fct.model')
     
    scores = evaluate(path, model)
    scores.to_csv(export_path, index=False)
    print(f"The scores have been saved to {export_path}")
    
    
    path = 'ACDC/testing/'
    export_path = 'results/fct_scores_test_2.csv'
     
    scores = evaluate(path, model)
    scores.to_csv(export_path, index=False)
    print(f"The scores have been saved to {export_path}")
    
                
        