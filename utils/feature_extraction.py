from skimage import measure
import scipy
import scipy.ndimage as morphology
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils.interface as interface
import torch
import os
import nibabel as nib
import scipy
from scipy import ndimage
import skimage
from skimage import feature


def heart_metrics(seg_3Dmap, voxel_size, classes=[3, 1, 2]):
    """
    Compute the volumes of each classes
    """
    # Loop on each classes of the input images
    volumes = []
    for c in classes:
        # Copy the gt image to not alterate the input
        seg_3Dmap_copy = np.copy(seg_3Dmap)
        seg_3Dmap_copy[seg_3Dmap_copy != c] = 0

        # Clip the value to compute the volumes
        seg_3Dmap_copy = np.clip(seg_3Dmap_copy, 0, 1)

        # Compute volume
        # volume = seg_3Dmap_copy.sum() * np.prod(voxel_size) / 1000.
        volume = seg_3Dmap_copy.sum() * np.prod(voxel_size)
        volumes += [volume]
    return volumes

def myocardial_thickness(label_obj, slices_to_skip=(0,0), myo_label=2):
    """
    Calculate myocardial thickness of mid-slices, excluding a few apex and basal slices
    since myocardium is difficult to identify
    """
#     label_obj = nib.load(data_path)
    myocardial_mask = (np.asanyarray(label_obj.dataobj)==myo_label)
    # pixel spacing in X and Y
    pixel_spacing = label_obj.header.get_zooms()[:2]
    assert pixel_spacing[0] == pixel_spacing[1]

    holes_filles = np.zeros(myocardial_mask.shape)
    interior_circle = np.zeros(myocardial_mask.shape)

    cinterior_circle_edge=np.zeros(myocardial_mask.shape)
    cexterior_circle_edge=np.zeros(myocardial_mask.shape)

    overall_avg_thickness= []
    overall_std_thickness= []
    for i in range(slices_to_skip[0], myocardial_mask.shape[2]-slices_to_skip[1]):
        holes_filles[:,:,i] = ndimage.morphology.binary_fill_holes(myocardial_mask[:,:,i])
        interior_circle[:,:,i] = holes_filles[:,:,i] - myocardial_mask[:,:,i]
        cinterior_circle_edge[:,:,i] = feature.canny(interior_circle[:,:,i])
        cexterior_circle_edge[:,:,i] = feature.canny(holes_filles[:,:,i])
        # patch = 64
        # utils.imshow(data_augmentation.resize_image_with_crop_or_pad(myocardial_mask[:,:,i], patch, patch), 
        #     data_augmentation.resize_image_with_crop_or_pad(holes_filles[:,:,i], patch, patch),
        #     data_augmentation.resize_image_with_crop_or_pad(interior_circle[:,:,i], patch,patch ), 
        #     data_augmentation.resize_image_with_crop_or_pad(cinterior_circle_edge[:,:,i], patch, patch), 
        #     data_augmentation.resize_image_with_crop_or_pad(cexterior_circle_edge[:,:,i], patch, patch), 
        #     title= ['Myocardium', 'Binary Hole Filling', 'Left Ventricle Cavity', 'Interior Contour', 'Exterior Contour'], axis_off=True)
        x_in, y_in = np.where(cinterior_circle_edge[:,:,i] != 0)
        number_of_interior_points = len(x_in)
        # print (len(x_in))
        x_ex,y_ex=np.where(cexterior_circle_edge[:,:,i] != 0)
        number_of_exterior_points=len(x_ex)
        # print (len(x_ex))
        if len(x_ex) and len(x_in) !=0:
            total_distance_in_slice=[]
            for z in range(number_of_interior_points):
                distance=[]
                for k in range(number_of_exterior_points):
                    a  = [x_in[z], y_in[z]]
                    a=np.array(a)
                    # print a
                    b  = [x_ex[k], y_ex[k]]
                    b=np.array(b)
                    # dst = np.linalg.norm(a-b)
                    dst = scipy.spatial.distance.euclidean(a, b)
                    # pdb.set_trace()
                    # if dst == 0:
                    #     pdb.set_trace()
                    distance = np.append(distance, dst)
                distance = np.array(distance)
                min_dist = np.min(distance)
                total_distance_in_slice = np.append(total_distance_in_slice,min_dist)
                total_distance_in_slice = np.array(total_distance_in_slice)

            average_distance_in_slice = np.mean(total_distance_in_slice)*pixel_spacing[0]
            overall_avg_thickness = np.append(overall_avg_thickness, average_distance_in_slice)

            std_distance_in_slice = np.std(total_distance_in_slice)*pixel_spacing[0]
            overall_std_thickness = np.append(overall_std_thickness, std_distance_in_slice)

    # print (overall_avg_thickness)
    # print (overall_std_thickness)
    # print (pixel_spacing[0])
    return (overall_avg_thickness, overall_std_thickness)


def ejection_fraction(volume_ED, volume_ES):    
    return (volume_ED - volume_ES)*100/volume_ED


def read_patient_cfg(path):
    """
    Reads patient data in the cfg file and returns a dictionary
    """
    patient_info = {}
    with open(os.path.join(path, "Info.cfg")) as info:
        patient_info["patient_id"] = path[-3:]
        for line in info:
            l = line.rstrip().split(": ")
            patient_info[l[0]] = l[1]
            
    return patient_info
 
def get_info_df(path):
    results = []
    for root, directories, files in os.walk(path):
            for file in files:
                if file.endswith(".cfg"):
                    patient_info = read_patient_cfg(root)
                    results.append(patient_info)

    return pd.DataFrame(results)            

def segment(nifti_img, affine, header, model):
    data_loader = interface.prepare_data(nifti_img)
    result = interface.model_output(model, data_loader)
    seg_nifti_file = nib.Nifti1Image(result, affine, header)   
    return seg_nifti_file
      
def get_ED_ES(path, img_ED, img_ED_seg, img_ES, img_ES_seg, model, hdr_ED, hdr_ES):
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    nifti_file = nib.load(img_path)
                    nifti_img = nifti_file.get_fdata()
                    affine = nifti_file.affine
                    header = nifti_file.header
                    segment_file = segment(nifti_img, affine, header, model)
                    if ("01" in file[-9:]) or ("04" in file[-9:]):
                        img_ED.append(nifti_file)
                        img_ED_seg.append(segment_file)
                        hdr_ED.append(header)
                    else:
                        img_ES.append(nifti_file)
                        img_ES_seg.append(segment_file)  
                        hdr_ES.append(header)
                                        

def features(ed_data, es_data, hdr_ed_patient, hdr_es_patient, pid, group, res):   
    
        
        ed_lv, ed_rv, ed_myo = heart_metrics(np.asanyarray(ed_data.dataobj),
                        hdr_ed_patient.get_zooms())
        es_lv, es_rv, es_myo = heart_metrics(np.asanyarray(es_data.dataobj),
                        hdr_es_patient.get_zooms())
        ef_lv = ejection_fraction(ed_lv, es_lv)
        ef_rv = ejection_fraction(ed_rv, es_rv)

        myo_properties = myocardial_thickness(es_data)
        es_myo_thickness_max_avg = np.amax(myo_properties[0])
        es_myo_thickness_std_avg = np.std(myo_properties[0])
        es_myo_thickness_mean_std = np.mean(myo_properties[1])
        es_myo_thickness_std_std = np.std(myo_properties[1])

        myo_properties = myocardial_thickness(ed_data)
        ed_myo_thickness_max_avg = np.amax(myo_properties[0])
        ed_myo_thickness_std_avg = np.std(myo_properties[0])
        ed_myo_thickness_mean_std = np.mean(myo_properties[1])
        ed_myo_thickness_std_std = np.std(myo_properties[1])


        heart_param = {'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
               'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
               'ES_MYO_MAX_AVG_T': es_myo_thickness_max_avg, 'ES_MYO_STD_AVG_T': es_myo_thickness_std_avg, 'ES_MYO_AVG_STD_T': es_myo_thickness_mean_std, 'ES_MYO_STD_STD_T': es_myo_thickness_std_std,
               'ED_MYO_MAX_AVG_T': ed_myo_thickness_max_avg, 'ED_MYO_STD_AVG_T': ed_myo_thickness_std_avg, 'ED_MYO_AVG_STD_T': ed_myo_thickness_mean_std, 'ED_MYO_STD_STD_T': ed_myo_thickness_std_std,}
        r=[]

        r.append(pid)
        r.append(heart_param['EDV_LV'])
        r.append(heart_param['ESV_LV'])
        r.append(heart_param['EDV_RV'])
        r.append(heart_param['ESV_RV'])
        r.append(heart_param['ED_MYO'])
        r.append(heart_param['ES_MYO'])
        r.append(heart_param['EF_LV'])
        r.append(heart_param['EF_RV'])
        r.append(ed_lv/ed_rv)
        r.append(es_lv/es_rv)
        r.append(ed_myo/ed_lv)
        r.append(es_myo/es_lv)
        # r.append(patient_data[pid]['Height'])
        # r.append(patient_data[pid]['Weight'])
        r.append(heart_param['ES_MYO_MAX_AVG_T'])
        r.append(heart_param['ES_MYO_STD_AVG_T'])
        r.append(heart_param['ES_MYO_AVG_STD_T'])
        r.append(heart_param['ES_MYO_STD_STD_T'])

        r.append(heart_param['ED_MYO_MAX_AVG_T'])
        r.append(heart_param['ED_MYO_STD_AVG_T'])
        r.append(heart_param['ED_MYO_AVG_STD_T'])
        r.append(heart_param['ED_MYO_STD_STD_T'])
        r.append(group)
        res.append(r)

        # df = pd.DataFrame(res, columns=HEADER)
        # return df

def extract_2(img_ED_seg, img_ES_seg, hdr_ED, hdr_ES,  patient_ID, category):
    res = []
    HEADER = ["Id", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
          "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]", "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
          "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", "ES[stdev(stdev(MWT|SA)|LA)]", 
          "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", "ED[stdev(stdev(MWT|SA)|LA)]", "Category"]

    for num in range(len(img_ED_seg)):
        ed_data = img_ED_seg[num]
        es_data = img_ES_seg[num]
        hdr_ed_patient = hdr_ED[num]
        hdr_es_patient = hdr_ES[num]
        pid = patient_ID[num]
        group = category[num]
        features(ed_data, es_data, hdr_ed_patient, hdr_es_patient, pid,group,  res)
    df = pd.DataFrame(res, columns=HEADER)    
    df.to_csv("./results/Cardiac_parameters-2.csv", index=False)
    return df
        

        


if __name__ == "__main__":
    train_path = 'ACDC/training' 
    test_path = 'ACDC/testing'
    # train_path = 'ACDC/check'
    export_path_test = 'results/test_patients_info_2.csv'
    export_path_train = 'results/train_patients_info_2.csv'
     
    
    train_info = get_info_df(train_path)
    test_info = get_info_df(test_path)
    # train_info.to_csv(export_path_train, index=False)
    # test_info.to_csv(export_path_test, index=False)  
    metaData = pd.concat([train_info, test_info], axis=0)
    patient_ID = metaData['patient_id'].values
    category = metaData['Group'].values
    height_patients = metaData['Height'].values
    weight_patients = metaData['Weight'].values
    
    img_ED, img_ED_seg, img_ES, img_ES_seg, hdr_ED, hdr_ES = [], [], [], [], [], []
    model = torch.load('models/fct.model')
        
    # model = UNet()
    # model.load_state_dict(torch.load("models/UNet.pt", map_location="cpu"))
        

    get_ED_ES(train_path, img_ED, img_ED_seg, img_ES, img_ES_seg, model, hdr_ED, hdr_ES)
    get_ED_ES(test_path, img_ED, img_ED_seg, img_ES, img_ES_seg, model, hdr_ED, hdr_ES)
    print("len each list:" ,  len(img_ED), len(img_ED_seg), len(img_ES), len(img_ES_seg))
    extract_2(img_ED_seg, img_ES_seg, hdr_ED, hdr_ES,  patient_ID)
    