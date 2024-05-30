import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from utils.data_utils import convert_masks
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from utils.data_utils import  extract_frames_test
import os
import joblib
import utils.feature_extraction as feature_extraction



def convert_mask_single(y):
    """
    Given one masks with many classes create one mask per class
    y: shape (w,h)
    """
    mask = np.zeros((4, y.shape[0], y.shape[1]))
    mask[0, :, :] = np.where(y == 0, 1, 0)
    mask[1, :, :] = np.where(y == 1, 1, 0)
    mask[2, :, :] = np.where(y == 2, 1, 0)
    mask[3, :, :] = np.where(y == 3, 1, 0)
    return mask

def visualize(image_raw,mask):
    """
    iamge_raw:gray image with shape [width,height,1]
    mask: segment mask image with shape [num_class,width,height]
    this function return an image using multi color to visualize masks in raw image
    """
    # Convert grayscale image to RGB
    image = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
    
#     image = image_raw
#     mask = mask.numpy()

    # Get the number of classes (i.e. channels) in the mask
    num_class = mask.shape[0]


    # Define colors for each class (using a simple color map)
    colors = []
    for i in range(1, num_class):  # skip first class (background)
        hue = int(i/float(num_class-1) * 179)
        color = np.zeros((1, 1, 3), dtype=np.uint8)
        color[0, 0, 0] = hue
        color[0, 0, 1:] = 255
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        colors.append(color)

    # Overlay each non-background class mask with a different color on the original image
    for i in range(1, num_class):
        class_mask = mask[i, :, :]
        class_mask = np.repeat(class_mask[:, :, np.newaxis], 3, axis=2)
        class_mask = class_mask.astype(image.dtype)
        class_mask = class_mask * colors[i-1]
        image = cv2.addWeighted(image, 1.0, class_mask, 0.5, 0.0)

    return image

def evaluate_model(model, dataloader):
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    results = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        y_pred = torch.argmax(outputs[2], axis=1)
        mask = convert_mask_single(y_pred[0, :, :].cpu().numpy())
        
        # Visualize the input image, ground truth mask, and predicted mask
        input_image = inputs[0].cpu().numpy().transpose(1, 2, 0)
        # convert into a single channel to visualize
        ground_truth_mask = torch.argmax(targets[0], dim=0)
        predicted_mask = y_pred.cpu().numpy().transpose(1, 2, 0)
        mask_with_image = visualize(input_image, mask)
        mask_with_image = (mask_with_image - mask_with_image.min()) / (mask_with_image.max()- mask_with_image.min()) *255
        # cv2.imwrite('steamlit.jpg',mask_with_image)    
        predicted_mask = predicted_mask * 64.0
        results.append({"input": input_image.astype(np.uint16),"predicted": np.where(predicted_mask == 256, 255, predicted_mask).astype(np.uint16), "both": mask_with_image.astype(np.uint16)})

        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1)
        # plt.title("Input Image")
        # plt.imshow(input_image, cmap='gray')

        # plt.subplot(1, 3, 2)
        # plt.title("Ground Truth Mask")        
        # plt.imshow(ground_truth_mask.cpu(), cmap='gray')
        
        # plt.subplot(1, 3, 3)
        # plt.title("Predicted Mask2")
        # plt.imshow(mask_with_image.astype(np.uint8))
        # plt.show()
    return results
        
def get_images(img, input_size=(224,224,1)):
    """
    given one .nii file and return all the frames in one list
    """
    all_imgs = []
    # img = nib.load(img).get_fdata()
    # img = img.get_fdata()
    for idx in range(img.shape[2]):
        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(f"raw_{idx}.png", img[:,:,idx].astype("float32"))
        all_imgs.append(i)
    all_imgs = np.expand_dims(all_imgs, axis=3)
    return [all_imgs, torch.empty((all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[2], 1), dtype=torch.float32)]
 
def model_output(model, dataloader):
    
    """
    return mask of an input file in format (10, 224, 224) to save in .nii file
    """
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)
    results = []
    
    """
    outputs[2],  torch.Size([1, 4, 224, 224])
    y_pred,  torch.Size([1, 224, 224])
    mask,  (4, 224, 224)
    predicted_mask  (224, 224, 1)
    """
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            
        y_pred = torch.argmax(outputs[2], axis=1)
        
        results.append(y_pred[0,:,:].cpu().numpy())
    results = np.array(results).transpose(1, 2, 0)
    return results
      
def prepare_data(img):
    """"
    input one image(nib.load) output data loader of that image file
    """
    acdc_data= get_images(img, input_size=(224, 224, 1))
    acdc_data[1] = convert_masks(acdc_data[1])
    acdc_data[0] = np.transpose(acdc_data[0], (0, 3, 1, 2)) # for the channels
    acdc_data[1] = np.transpose(acdc_data[1], (0, 3, 1, 2)) # for the channels
    acdc_data[0] = torch.Tensor(acdc_data[0]) # convert to tensors
    acdc_data[1] = torch.Tensor(acdc_data[1]) # convert to tensors
    acdc_data = TensorDataset(acdc_data[0], acdc_data[1])
    test_loader = DataLoader(acdc_data, batch_size=1, num_workers=2)
    return test_loader
    
def predict(img, model):
    """"
    used in streamlit app to predict the input img, return a list to plot in the app
    """
    test_loader = prepare_data(img)
    results = evaluate_model(model, test_loader)
    return results

def most_frequent(nums):
    return max(set(nums), key=nums.count)

def segment(inputs, model):
    
    """
    return mask of an input file in format (10, 224, 224) to save in .nii file
    """
    results = []
    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size*2,1, 224, 224)
    with torch.no_grad():
        outputs = model(inputs)
        
    print("outputs.shape", outputs[2].shape)
    y_pred = torch.argmax(outputs[2], axis=1)
    y_pred = y_pred.reshape(batch_size, 2, 224, 224)
    print("y_pred.shape", y_pred.shape)
    results = np.array(y_pred.cpu(), dtype="float32")
    return results

def get_vae_bottleneck(ED_ES_seg, vae):
    with torch.no_grad():
        X_train = vae.bottle_neck(torch.from_numpy(ED_ES_seg.reshape(ED_ES_seg.shape[0]*2, 1, 224, 224)))
        X_train = X_train.reshape(ED_ES_seg.shape[0], 2*16).numpy()
    return X_train

def pipeline(ED_ES, seg_model, vae, hdr_ed_patient, hdr_es_patient, affine_ed, affine_es):
    HEADER = ["Id", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
                                    "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]", "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
                                    "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", "ES[stdev(stdev(MWT|SA)|LA)]", 
                                    "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", "ED[stdev(stdev(MWT|SA)|LA)]", "Category"]

    scaler = joblib.load("transformers/robustscaler.joblib")
    device = torch.device("cuda")
    ED_ES= ED_ES.to(device)
    print("ED_ES.shape", ED_ES.shape, "ED_ES.dtype", ED_ES.dtype, "ED_ES.min()", ED_ES.min(), " ED_ES.max()", ED_ES.max())
    batch_seg =  segment(ED_ES, seg_model) # -> (batch, 2, 224, 224)
    batch_seg_norm = batch_seg.copy() / 3.
    
    ED_seg = batch_seg[:,0,:,:].astype("int64").transpose(1, 2, 0)
    ES_seg = batch_seg[:,1,:,:].astype("int64").transpose(1, 2, 0)
        
    nifti_file_ED = nib.Nifti1Image(ED_seg, affine_ed, hdr_ed_patient)
    nifti_file_ES = nib.Nifti1Image(ES_seg, affine_es, hdr_es_patient)
    
    patient = []
    feature_extraction.features(nifti_file_ED, nifti_file_ES, hdr_ed_patient, hdr_es_patient, "", "", patient)
    patient= pd.DataFrame(patient, columns=HEADER).rename({0:"Patient"}, axis=0)   
    patient = patient.drop(["Id", "Category"], axis=1)
    f = np.array(patient)
    f_extracted = []
    for i in range(batch_seg.shape[0]):
        f_extracted.append(f)
    
    f_extracted = np.array(f_extracted)[:, 0,:]
    print("f_extracted.shape", f_extracted.shape)
    f_extracted = scaler.transform(f_extracted)
        
    print("batch_seg.shape", batch_seg.shape, "batch_seg.dtype", batch_seg.dtype, "batch_seg.min()", batch_seg.min(), " batch_seg.max()", batch_seg.max())
    batch_vae = get_vae_bottleneck(batch_seg_norm, vae) # -> (batch, 64)
    print("batch_vae.shape", batch_vae.shape)
    batch_X_f = np.concatenate([batch_vae, f_extracted], axis=1)
    print("batch_X_f.shape", batch_X_f.shape)        
    return batch_X_f, patient



def classify(img_ED, img_ES, seg_model, vae, classification_model, hdr_ed_patient, hdr_es_patient, affine_ed, affine_es):
    categories = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']
    
    img_ED, img_ES, y = extract_frames_test(img_ED, img_ES, [])
    ED_ES = np.stack([img_ED, img_ES], axis=1)
    ED_ES = torch.tensor(ED_ES).float()
    # manual_features = manual_features[:,0,:]
    print("[!] After extract frames:")
    print("[!] ED.shape: ", img_ED.shape, " ES.shape: ", img_ES.shape, " y.shape: ",y.shape)
      
    deep_manual_features, manual_features_df = pipeline(ED_ES, seg_model, vae, hdr_ed_patient, hdr_es_patient, affine_ed, affine_es)
    
    print("manual_features.shape", manual_features_df.shape)
    print("deep_manual_features.shape",deep_manual_features.shape)
    
    preds = classification_model.predict(deep_manual_features)
    print("preds,", preds)
    preds = list(preds)
    pred_class = most_frequent(preds)

    return categories[pred_class], manual_features_df



def confidence(outputs, pred_class):
    out = outputs.softmax(dim=1)
    conf = 0
    for i in range(out.shape[0]):
        conf += out[i][pred_class].item()
    conf = conf / out.shape[0]
    return conf 

def save_predictions(path, model):
    for root, directories, files in os.walk(path):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    out_path = root + "/" + file.split(".nii.gz")[0] +"_sg" + ".nii.gz"
                    print(out_path)
                    img = nib.load(img_path).get_fdata()
                    data_loader = prepare_data(img)
                    result = model_output(model, data_loader)
                    result = result.astype("float64")
                    affine = np.eye(4)
                    nifti_file = nib.Nifti1Image(result, affine)
                    nib.save(nifti_file, out_path) # Here you put the path + the extionsion 'nii' or 'nii.gz'
       
if __name__ == "__main__":
    model = torch.load('models/fct.model')
    save_predictions("ACDC/training", model)
    save_predictions("ACDC/testing", model)