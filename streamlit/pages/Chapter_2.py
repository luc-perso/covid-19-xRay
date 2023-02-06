# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:15:36 2023

@author: PC
"""

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os
#st.set_page_config(layout="wide")
#__________________________________ SIDEBAR ________________________________________
st.sidebar.markdown("# Model comparison")


#__________________________________ PAGE CONTENT ________________________________________

root = "E:\covid-xray\streamlit\multipage"
st.markdown("# Model comparison")
st.markdown("We tested the performance of different model architectures on original and masked dataset for classifiing X-ray images as Normal, COVID-19 and non-COVID.\
        Select the different models tested, to explore their architecture and the training results.")

model_pic = st.selectbox("View model architecture",('Linear1', 'Linear2', 'Linear3', 'VGG16', 'ResNet101', 'InceptionV3', 'Xception'))
st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\models\{}.png'.format(model_pic))),
         model_pic, 700, 600)

sel = st.selectbox("Results of which dataset do you want to display?",('All results', 'Results on masked pre-processed dataset', 'Results on original pre-processed dataset'))

if sel == "All results":
    
    models = st.multiselect('Which models trained on original data  would you like to compare?',
                   ['Linear1', 'Linear2', 'Linear3', 'VGG16', 'ResNet101', 'InceptionV3', 'Xception'],
                   ['Linear1', 'Linear2', 'Linear3'])
    models_mask = st.multiselect('Which models trained on masked data would you like to compare?',
                   ['Linear1', 'Linear2', 'Linear3', 'VGG16', 'ResNet101', 'InceptionV3', 'Xception'],
                   ['Linear1', 'Linear2', 'Linear3'])
    
    st.text("Training History")
    width = st.sidebar.slider("plot size(width)", 400, 1000, 400)
    data = pd.read_csv(os.path.join(root,"pages\data\Chapter2\history_not_masked.csv"), index_col = [0,1])
    data_masked = pd.read_csv(os.path.join(root,"pages\data\Chapter2\history_masked.csv"), index_col = [0,1])
    plotted_data = plt.Figure()
    ax = plotted_data.add_subplot(111)
    ax.set_title("Training history on masked data for selected models")
    for model in models:
        ax.plot(data.loc[model, :].index, data.loc[model, "acc"], label = model)
    for model in models_mask:
        ax.plot(data_masked.loc[model, :].index, data_masked.loc[model, "acc"], label = model+"_mask")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    
    buf = BytesIO()
    plotted_data.savefig(buf, format="png", bbox_inches = "tight")
    st.image(buf, width=width, use_column_width=False)
    
    st.text("Tabular Results")
    st.dataframe(pd.read_csv(os.path.join(root,"pages\data\Chapter2\\tab_results.csv"), index_col=(0,1), header = [0,1]).loc[(["masked","not_masked"],models),:])
    
    st.text("heatmap")
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\heatmap_linear_edit.png')),
             "Heatmaps for simple linear models for original dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\heatmap_transfer_edit.png')),
             "Heatmaps for transfer learning models for original dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\heatmap_linear_edit.png')),
             "Heatmaps for simple linear models for masked dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\heatmap_transfer_edit.png')),
             "Heatmaps for transfer learning models for masked dataset", 500, 600)
    
    
    st.text("GradCam")
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\gradcam_linear_edit.png')),
             "GradCams for simple linear models for original dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\gradcam_transfer_edit.png')),
             "GradCams for transfer learning models for original dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\gradcam_linear_edit.png')),
             "GradCams for simple linear models for masked dataset", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\gradcam_transfer_edit.png')),
             "GradCams for transfer learning models for masked dataset", 500, 600)





    
# presenting MASKED results    
elif sel == 'Results on masked pre-processed dataset':
    models = st.multiselect('Which model would you like to compare',
                   ['Linear1', 'Linear2', 'Linear3', 'VGG16', 'ResNet101', 'InceptionV3', 'Xception'],
                   ['Linear1', 'Linear2', 'Linear3'])
    
    st.text("Training History")
    width = st.sidebar.slider("plot size(width)", 400, 1000, 400)
    data = pd.read_csv(os.path.join(root,"pages\data\Chapter2\history_masked.csv"), index_col = [0,1])
    plotted_data = plt.Figure()
    ax = plotted_data.add_subplot(111)
    ax.set_title("Training history on masked data for selected models")
    for model in models:
        ax.plot(data.loc[model, :].index, data.loc[model, "acc"], label = model)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    
    buf = BytesIO()
    plotted_data.savefig(buf, format="png", bbox_inches = "tight")
    st.image(buf, width=width, use_column_width=False)
    
    st.text("Tabular Results")
    st.dataframe(pd.read_csv(os.path.join(root,"pages\data\Chapter2\\tab_results.csv"), index_col=(0,1), header = [0,1]).loc[("masked",models),:])
    
    st.text("Heatmap")
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\heatmap_linear_edit.png')),
             "Heatmaps for simple linear models", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\heatmap_transfer_edit.png')),
             "Heatmaps for transfer learning models", 500, 600)
    
    st.text("GradCam")
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\gradcam_linear_edit.png')),
             "GradCams for simple linear models", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\masked\gradcam_transfer_edit.png')),
             "GradCams for transfer learning models", 500, 600)
    





# presenting NOT_MASKED results
elif sel == 'Results on original pre-processed dataset':
    models = st.multiselect('Which model would you like to compare',
                   ['Linear1', 'Linear2', 'Linear3', 'VGG16', 'ResNet101', 'InceptionV3', 'Xception'],
                   ['Linear1', 'Linear2', 'Linear3'])
    
    st.text("Training History")
    width = st.sidebar.slider("plot size(width)", 400, 1000, 400)
    data = pd.read_csv(os.path.join(root,"pages\data\Chapter2\history_not_masked.csv"), index_col = [0,1])
    plotted_data = plt.Figure()
    ax = plotted_data.add_subplot(111)
    ax.set_title("Training history on original data for selected models")
    for model in models:
        ax.plot(data.loc[model, :].index, data.loc[model, "acc"], label = model)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    
    buf = BytesIO()
    plotted_data.savefig(buf, format="png", bbox_inches = "tight")
    st.image(buf, width=width, use_column_width=False)
    
    st.text("Tabular Results")
    st.dataframe(pd.read_csv(os.path.join(root,"pages\data\Chapter2\\tab_results.csv"), index_col=(0,1), header = [0,1]).loc[("not_masked",models),:])
    
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\heatmap_linear_edit.png')),
             "Heatmaps for simple linear models", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\heatmap_transfer_edit.png')),
             "Heatmaps for transfer learning models", 500, 600)
    
    st.text("GradCam")
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\gradcam_linear_edit.png')),
             "GradCams for simple linear models", 500, 600)
    st.image(Image.open(os.path.join(root,'pages\data\Chapter2\imgs\\not_masked\gradcam_transfer_edit.png')),
             "GradCams for transfer learning models", 500, 600)