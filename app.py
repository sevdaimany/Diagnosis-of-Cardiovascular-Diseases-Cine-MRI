import streamlit as st
import numpy as np
import  utils.interface as interface
import tempfile
import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import utils.feature_extraction as feature_extraction
import pandas as pd
from utils.unet import UNet
import plotly.graph_objects as go
from io import BytesIO
import imageio
import base64
import joblib
from utils.vae.main import VAE


def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    torch.save(model, model_path)
    return model_path

def main():
    
    st.sidebar.title("Settings")
    # st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ["Segmentation", "Feature Extraction", "Classification", "Evaluation", ])
    
    
    if app_mode == 'About App':
        
        
        
        st.header("Deep Learning-Based Diagnosis of Cardiovascular Diseases using 4D Cine MRI")
        
        st.header("Introduction")
        
        
        # st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
        # st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)  
        # st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)
        # st.markdown("""<p>🔍Some of these modifications are:<br>
        #             &#x2022; Introducing a new backbone network, Darknet-53,<br>
        #             &#x2022; Introducing a new anchor-free detection head. This means it predicts directly the center of an object instead of the offset from a known anchor box.<br>
        #             &#x2022; and a new loss function.<br></p>""", unsafe_allow_html=True)
        
        # st.markdown("""<p>🎊One of the key advantages of YOLOv8 is its versatility. It not only supports object detection but also offers out-of-the-box support for classification and segmentation tasks. This makes it a powerful tool for various computer vision applications.<br><br>
        #             ✨In this project, we will focus on three major computer vision tasks that YOLOv8 can be used for: <b>classification</b>, <b>detection</b>, and <b>segmentation</b>. We will explore how YOLOv8 can be applied in the field of medical imaging to detect and classify various anomalies and diseases🧪💊.</p>""", unsafe_allow_html=True)
        
        # st.markdown("""<p>We hope you find this project informative and inspiring.💡 Let's dive into the world of YOLOv8 and discover how easy it is to use it!🥁🎆</p>""", unsafe_allow_html=True)
    
    elif app_mode == "Segmentation":
        
        # Cardiac Abnormality Diagnostic System using MRI
        html_temp = """ 
        <div style="background-color:orange ;padding:7px;margin-bottom:20px">
        <h2 style="color:black;text-align:center;"><b>Cine MRI Segmentation<b></h2>
        </div>
        """ 
        st.markdown(html_temp,unsafe_allow_html=True)

        st.sidebar.markdown("----")
        
        uploaded_files = st.sidebar.file_uploader("Upload File (NIfTI)", type=["nii", "nii.gz"], key=0)
        model = torch.load('models/fct.model')
        unet = UNet()
        unet.load_state_dict(torch.load("models/UNet.pt", map_location="cpu"))
        
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                bytes_data = uploaded_files.read()
                file_path = os.path.join(temp_dir, uploaded_files.name)
                with open(file_path, 'wb') as f:
                    f.write(bytes_data)
                file_name = "seg_" + uploaded_files.name    
    
                image_np = load_nifti_file(file_path, model, unet, file_name)
                axial_slice_num = st.slider(' ', 0, image_np.shape[2] - 1, 0, key="axial_slider")
                
                st.subheader("FCT model results:")
                display_segments(axial_slice_num, "seg_results")
                st.subheader("Unet model results:")
                display_segments(axial_slice_num, "seg_results_unet")
                
          
    elif app_mode == "Evaluation":
        
        html_temp = """ 
        <div style="background-color:orange ;padding:7px; margin-bottom:20px">
        <h2 style="color:black;text-align:center;"><b>Analysis of the models<b></h2>
        </div>
        """ 
        st.markdown(html_temp,unsafe_allow_html=True)

        st.subheader("1) Segmentation model:")
        
        st.image('imgs/predicted_vs_gt.png', caption='predicted and groud truth masks')
        col1, col2= st.columns(2)
        scores = pd.read_csv('./results/dice_train.csv')
        patients = pd.read_csv('./results/train_patients_info.csv')
        merged = pd.merge(scores, patients, on='patient_id')
        groups = merged[['Group', 'dice_avg', 'dice_lv', 'dice_myo', 'dice_rv']]
        df_train = groups.groupby("Group").mean()[["dice_avg", "dice_lv", "dice_rv", "dice_myo"]]
        display_barchart_dice(df_train, col1, "training dataset")
        

        scores = pd.read_csv('./results/dice_test.csv')
        patients = pd.read_csv('./results/test_patients_info.csv')
        merged = pd.merge(scores, patients, on='patient_id')
        groups = merged[['Group', 'dice_avg', 'dice_lv', 'dice_myo', 'dice_rv']]
        df_test = groups.groupby("Group").mean()[["dice_avg", "dice_lv", "dice_rv", "dice_myo"]]
        display_barchart_dice(df_test, col1, "testing dataset")
        
        col1, col2= st.columns(2)
        with col1:
            st.write("Mean Dice Values by Category on training dataset")
            st.write(df_train)

        with col2:
            st.write("Mean Dice Values by Category on test dataset")
            st.write(df_test)

        
        st.subheader("2) VAE model:")
        total_loss_train =  pd.read_csv("results/variational_autoencoder/train_totalloss_epoch.csv")
        cce_loss_train =  pd.read_csv("results/variational_autoencoder/train_cce_epoch.csv")
        
        total_loss_val =  pd.read_csv("results/variational_autoencoder/val_totalloss_epoch.csv")
        accuracy_train =  pd.read_csv("results/variational_autoencoder/train_accuracy_epoch.csv")
        accuracy_val =  pd.read_csv("results/variational_autoencoder/val_accuracy_epoch.csv")
        # cce_loss_val =  pd.read_csv("results/variational-autoencoder/val_loss_step.csv")
        
        col1, col2 = st.columns(2)
        display_chart_model(total_loss_train, col1, "total loss train", "Epoch")
        display_chart_model(cce_loss_train, col2, "categorical cross-entropy train", "Epoch")
        
        display_chart_model(total_loss_val, col2, "total loss val", "Epoch")
        display_chart_model(accuracy_val, col1, "accuracy val", "Epoch")
        display_chart_model(accuracy_train, col1, "accuracy train", "Epoch")
        
        st.image('imgs/vae-train.png', caption='the reconstructed image in different epochs')
        
        
        st.subheader("3) Voting Classifier:")
        
        st.image('imgs/confusion-train.png')
        st.image('imgs/confusion-test.png')
        st.image('imgs/learning-curve.png')
    
        
    
        
    
    elif app_mode == "Feature Extraction":
        
        html_temp = """ 
        <div style="background-color:orange ;padding:7px; margin-bottom:20px">
        <h2 style="color:black;text-align:center;"><b>Extract Cardiac Features<b></h2>
        </div>
        """ 
        st.markdown(html_temp,unsafe_allow_html=True)

        model = torch.load('models/fct.model')
        features_seg = pd.read_csv("results/Cardiac_parameters.csv")
        normal = features_seg[features_seg["Category"] == "NOR"]
        normal  = normal.drop(["Id", "Category"], axis=1)

        st.sidebar.title("Upload Files")
        ED = st.sidebar.file_uploader("Upload File ED (NIfTI)", type=["nii.gz"], key=1)
        ES = st.sidebar.file_uploader("Upload File ES (NIfTI)", type=["nii.gz"], key=2)

        if st.sidebar.button("Extract"):
            if ED is not None and ES is not None:
                    # Load NIfTI files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file1:
                    temp_file1.write(ED.read())
                    temp_path1 = temp_file1.name
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file2:
                    temp_file2.write(ES.read())
                    temp_path2 = temp_file2.name
                img1 = nib.load(temp_path1)
                img2 = nib.load(temp_path2)
                
                with st.spinner("Model is Segmenting the input files, Please wait.."):
                    
                    file_name = ED.name    
                    ED_seg = load_nifti_ED_ES(img1, model) 
                    ES_seg = load_nifti_ED_ES(img2, model) 
                    
                    # patient = feature_extraction.extract([img1], [ED_seg], [img2], [ES_seg])
                    patient = []
                    hdr_ed_patient = img1.header
                    hdr_es_patient = img2.header
                    HEADER = ["Id", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
                                "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]", "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
                                "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", "ES[stdev(stdev(MWT|SA)|LA)]", 
                                "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", "ED[stdev(stdev(MWT|SA)|LA)]", "Category"]

                    feature_extraction.features(ED_seg, ES_seg, hdr_ed_patient, hdr_es_patient, "", "", patient)
                    patient= pd.DataFrame(patient, columns=HEADER).rename({0:"Patient"}, axis=0)   
    
                    patient = patient.drop(["Id", "Category"], axis=1)
                        
                    anomalies = detect_anomalies(patient, normal)
                    patient = add_normal_stats_to_patient(patient, normal)

                        # df.to_csv("feature_extraction_streamlit.csv", index=False)
                    st.subheader("Features dataframe:")
                    display_df_with_anomalies(patient, anomalies)
            else:
                st.warning("Please upload both files.")

                    
                        
    elif app_mode == "Classification":
        
        
            html_temp = """ 
            <div style="background-color:orange ;padding:7px;margin-bottom:20px">
            <h2 style="color:black;text-align:center;"><b>Diagnose Cardiac Abnormality<b></h2>
            </div>
            """ 
            st.markdown(html_temp,unsafe_allow_html=True)

            model_classification = joblib.load('models/classifier.pkl')
            seg_model = torch.load('models/fct.model')
            seg_model.eval()
            seg_model.to("cuda")
    
            # input_dim = 224
            # latent_dim = 16
            # resume = "218-250"
            # start_epoch = 0
            # end_epoch = 0
            # vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
            # if resume is not None:
            #     start_epoch = int(resume.split("-")[0])
            #     end_epoch = int(resume.split("-")[1])
            # #     vae.load_state_dict(torch.load(f"./utils/vae/Epoch[{resume}].pth"))
            #     vae.load_state_dict(torch.load(f"./models/old_models/vae-1.pth"), strict=False)
            # save_model(vae, "vautoencoder.model")

            vae = torch.load("models/vautoencoder.model")
            vae.eval()
            
            
            # File inputs
            st.sidebar.title("Upload Files")
            ED = st.sidebar.file_uploader("Upload File ED (NIfTI)", type=["nii.gz"], key=3)
            ES = st.sidebar.file_uploader("Upload File ES (NIfTI)", type=["nii.gz"], key=4)

            if st.sidebar.button("Classify"):
                if ED is not None and ES is not None:
                    # Load NIfTI files
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file1:
                        temp_file1.write(ED.read())
                        temp_path1 = temp_file1.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file2:
                        temp_file2.write(ES.read())
                        temp_path2 = temp_file2.name
                    img1 = nib.load(temp_path1)
                    img2 = nib.load(temp_path2)
                    # Extract voxel data (assuming they represent height and weight)
                    data1 = img1.get_fdata()
                    data2 = img2.get_fdata()
                    print("data1.shape", data1.shape)
                    print("data2.shape", data2.shape)
                    
                    print("np.concatenate([data1, data2],axis=2).shape", np.concatenate([data1, data2],axis=2).transpose(2, 0, 1).shape)
                    file_name = ED.name    
                    hdr_ed_patient = img1.header
                    hdr_es_patient = img2.header
                    affine_ed = img1.affine
                    affine_es = img2.affine
                
                            
                    with st.spinner("Model is predicting the input files, Please wait.."):
                        classification , patient= interface.classify(data1, data2, seg_model, vae, model_classification, hdr_ed_patient, hdr_es_patient, affine_ed, affine_es)
                    
                    video_array = np.concatenate([data1, data2],axis=2).transpose(2, 0, 1).astype("uint8")

                    gif_bytes = display_video(video_array)
                    gif_base64 = base64.b64encode(gif_bytes).decode("utf-8")
                    
                    st.dataframe(patient, use_container_width=True,  height = 100)
                    
                    col1, col2= st.columns(2)
                    with col1:
                        st.markdown(f'<img src="data:image/gif;base64,{gif_base64}" height="400" width= "400" style="padding:10%" />', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<p style="color:orange;font-size:28px;font-weight: bold;padding-top:50%;padding-left:10%">Predicted class: {classification}</p>', unsafe_allow_html=True)
                else:
                    st.warning("Please upload both files.")


def display_video(array, fps=30):
    rgb_array = np.stack((array,) * 3, axis=-1)

    # Write frames to memory as a GIF
    with BytesIO() as gif_bytes:
        imageio.mimsave(gif_bytes, rgb_array, format='GIF', duration=0.1, loop=0)
        gif_bytes.seek(0)
        return gif_bytes.getvalue()

def display_barchart_dice(df, col, caption):
    with col:
        # st.bar_chart(df)
        fig = go.Figure()

        for col in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[col],
                name=col
            ))

        fig.update_layout(
            barmode='group',  # Group bars
            xaxis_title='Category',
            yaxis_title='Mean Dice Value',
            title=f'Mean Dice Values by Category on {caption}'
        )
        st.plotly_chart(fig)



    
def display_chart_model(df, col, caption, x_axis):
    with col:
        df = df[["Step", "Value"]]
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Step'],
            y=df['Value'],
            mode='lines'
        ))

        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title='Value',
            title=caption
        )
        st.plotly_chart(fig, use_container_width=True)

        

        
def display_segments(axial_slice_num, key):
    col1, col2, col3 = st.columns(3)
    seg_results = st.session_state[key] 
    with col1:
        st.subheader("Input image")
        st.image(seg_results[axial_slice_num]["input"], caption='Image 1', channels="GRAY", clamp=True, use_column_width=True)
    with col2:
        st.subheader("Predicted mask")
        st.image(seg_results[axial_slice_num]["predicted"], caption='Image 2', channels="GRAY", clamp=True,  use_column_width=True)
    with col3:
        st.subheader("Predicted image")
        st.image(seg_results[axial_slice_num]["both"], caption='Image 3',clamp=True, channels='RGB',  use_column_width=True)
    
def display_df_with_anomalies(patient, anomaly_dict):
    styled_df = patient.copy().transpose()
    
    def highlight_anomalies(col):
        if col.name in anomaly_dict and anomaly_dict[col.name]:
            return ['background-color: #E97451'] * len(col)
        else:
            return ['background-color: white'] * len(col)
    
    styled_df = styled_df.style.apply(highlight_anomalies, axis=1)
    st.dataframe(styled_df, use_container_width=True, height = int(35.2*(patient.shape[1]+1))) 
     
                      
def add_normal_stats_to_patient(patient, normal):    
    normal_stats = normal.describe().loc[['mean', 'std']]
    
    for col in patient.columns:
        patient.loc['mean_normal', col] = normal_stats.loc['mean', col]
        patient.loc['std_normal', col] = normal_stats.loc['std', col]
    
    return patient
 
def detect_anomalies(patient, normal):
    anomalies = {}
    for col in normal.columns:
        mean = normal[col].mean()
        std = normal[col].std()
        if (patient[col].item() > mean + (2 * std)) or (patient[col].item() < mean - (2 * std)):
            anomalies[col] = True
        else:
            anomalies[col] = False
    return anomalies  

def load_nifti_ED_ES(ED_nifti_file, model):
        # ED_nifti_file = nib.load(filepath)
        ED_nifti_img = ED_nifti_file.get_fdata()
                
        data_loader = interface.prepare_data(ED_nifti_img)
        result = interface.model_output(model, data_loader)
        affine = ED_nifti_file.affine
        header = ED_nifti_file.header
        nifti_file = nib.Nifti1Image(result, affine, header) 
        return nifti_file
              
def load_nifti_file(filepath, model,unet, session_key):
    
    if session_key not in st.session_state:
        nifti_img = nib.load(filepath)
        image_np = np.asanyarray(nifti_img.dataobj)
        st.session_state[session_key] = image_np
        
        with st.spinner("Model is predicting the input file, Please wait.."):
            st.session_state["seg_results"] = interface.predict(image_np, model)
            st.session_state["seg_results_unet"] = interface.predict(image_np, unet)
            
                
    return st.session_state[session_key]        
       
def plot_slice(slice, size=(4, 4)):
    # Adjust the figure size for consistent viewer sizes
    fig, ax = plt.subplots(figsize=size)
    # Calculate the square canvas size
    canvas_size = max(slice.shape)
    canvas = np.full((canvas_size, canvas_size), fill_value=slice.min(), dtype=slice.dtype)
    # Center the image within the canvas
    x_offset = (canvas_size - slice.shape[0]) // 2
    y_offset = (canvas_size - slice.shape[1]) // 2
    canvas[x_offset:x_offset+slice.shape[0], y_offset:y_offset+slice.shape[1]] = slice
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')
    # canvas = np.rot90(canvas)

    ax.imshow(canvas, cmap='gray')
    ax.axis('off')
    return fig

     


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
        

