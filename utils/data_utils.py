import os
import cv2
import nibabel as nib
import numpy as np
from monai import data, transforms
from monai.data import load_decathlon_datalist
import re

def get_acdc(path,input_size=(224,224,1)):
    """
    Read images and masks for the ACDC dataset
    """
    all_imgs = []
    all_gt = []
    all_header = []
    all_affine = []
    info = []
    print("hi!")
    for root, directories, files in os.walk(path):
        files = sorted(files, key=lambda x: tuple(int(i) for i in re.findall('\d+', x)[1:]))
        for file in files:
            if (".gz" and "frame" in file) and ("_sg" not in file):
                print(file)
                if "_gt" not in file:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    all_header.append(nib.load(img_path).header)
                    all_affine.append(nib.load(img_path).affine)                   
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_imgs.append(i)
                        
                else:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_gt.append(i)
            

    data = [all_imgs, all_gt, info]                  
 
      
    data[0] = np.expand_dims(data[0], axis=3)
    if path[-9:] != "true_test":
        data[1] = np.expand_dims(data[1], axis=3)
    
    return data, all_affine, all_header

def get_classification_data(path, classes, features, input_size=(224,224,1)):
    img_ED = []
    img_ES = []
    y = []
    manual_features = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    nifti_file = nib.load(img_path)
                    nifti_img = nifti_file.get_fdata()
                    patient_id = int(file.split("_")[0][-3:])
                    patient_class = classes.loc[patient_id]["Category"]
                    if ("01" in file[-9:]) or ("04" in file[-9:]):
                    # if ("01" in file[-12:]) or ("04" in file[-12:]):
                        img_ED.append(nifti_img)
                        y.append(patient_class)
                        manual_features.append(features.loc[patient_id])
                    else:
                        img_ES.append(nifti_img)
    return img_ED, img_ES, y, manual_features


def get_classification_data2(path, classes, features, 
    img_ED, img_ES, y, manual_features, input_size=(224,224,1)):
    for root, dirs, files in os.walk(path):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    nifti_file = nib.load(img_path)
                    nifti_img = nifti_file.get_fdata()
                    patient_id = int(file.split("_")[0][-3:])
                    # print("patient_id ", patient_id)
                    patient_class = classes.loc[patient_id]["Category"]
                    if ("01" in file[-9:]) or ("04" in file[-9:]):
                      # print("ED file: ", file)
                      img_ED.append(nifti_img)
                      y.append(patient_class)
                      manual_features.append(features.loc[patient_id])
                    else:
                      img_ES.append(nifti_img)
  
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def extract_frames(ED, ES, manual_features, y, input_size=(224, 224)):
  ED_extracted = []
  ES_extracted = []
  f_extracted = []
  y_extracted = []
  for file_patient in range(y.shape[0]):
    nifti_img_ED = ED[file_patient]
    nifti_img_ES = ES[file_patient]

    for idx in range(nifti_img_ED.shape[2]):
        i = cv2.resize(nifti_img_ED[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST) / 255.
        j = cv2.resize(nifti_img_ES[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST) / 255.
        
        # i = crop_center(i, 128, 128)
        # j = crop_center(j, 128, 128)
        patient_class = y[file_patient]
        patient_feature = manual_features[file_patient]
        ED_extracted.append(i)
        ES_extracted.append(j)
        y_extracted.append(patient_class)
        f_extracted.append(patient_feature)
  return np.array(ED_extracted), np.array(ES_extracted), np.array(f_extracted), np.array(y_extracted)

                    
def extract_frames_test(ED, ES, y, input_size=(224, 224)):
  ED_extracted = []
  ES_extracted = []
#   f_extracted = []
  y_extracted = []
  for idx in range(ED.shape[2]):
        i = cv2.resize(ED[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
        j = cv2.resize(ES[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)        
        ED_extracted.append(i)
        ES_extracted.append(j)
        y_extracted.append(y)
        # f_extracted.append(manual_features)
  return np.array(ED_extracted), np.array(ES_extracted), np.array(y_extracted)



def convert_masks(y, data="acdc"):
    """
    Given one masks with many classes create one mask per class
    """
    
    if data == "acdc":
        # initialize
        masks = np.zeros((y.shape[0], y.shape[1], y.shape[2], 4))
        
        for i in range(y.shape[0]):
            masks[i][:,:,0] = np.where(y[i]==0, 1, 0)[:,:,-1] 
            masks[i][:,:,1] = np.where(y[i]==1, 1, 0)[:,:,-1] 
            masks[i][:,:,2] = np.where(y[i]==2, 1, 0)[:,:,-1] 
            masks[i][:,:,3] = np.where(y[i]==3, 1, 0)[:,:,-1]
            
    elif data == "synapse":
        masks = np.zeros((y.shape[0], y.shape[1], y.shape[2], 9))
        
        for i in range(y.shape[0]):
            masks[i][:,:,0] = np.where(y[i]==0, 1, 0)[:,:,-1]
            masks[i][:,:,1] = np.where(y[i]==1, 1, 0)[:,:,-1]  
            masks[i][:,:,2] = np.where(y[i]==2, 1, 0)[:,:,-1]  
            masks[i][:,:,3] = np.where(y[i]==3, 1, 0)[:,:,-1]  
            masks[i][:,:,4] = np.where(y[i]==4, 1, 0)[:,:,-1]  
            masks[i][:,:,5] = np.where(y[i]==5, 1, 0)[:,:,-1]  
            masks[i][:,:,6] = np.where(y[i]==6, 1, 0)[:,:,-1]  
            masks[i][:,:,7] = np.where(y[i]==7, 1, 0)[:,:,-1]  
            masks[i][:,:,8] = np.where(y[i]==8, 1, 0)[:,:,-1]  
            
    else:
        print("Data set not recognized")
        
    return masks

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