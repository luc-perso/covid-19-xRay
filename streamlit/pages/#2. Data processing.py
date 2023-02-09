# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:14:37 2023

@author: PC
"""
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os
#st.set_page_config(layout="wide")
#__________________________________ SIDEBAR ________________________________________
st.sidebar.markdown("# 2. Data processing")


#__________________________________ PAGE CONTENT ________________________________________

root = os.path.dirname(os.path.realpath(__file__))

st.markdown("# 2. Data processing")

##############################################################################

st.header("Data balance")
dataset = st.selectbox('Choose segmentation data',['infection segmentation data', 'lung segmentation data' ])


sets=['train', 'test', 'val']
cov1 =[1864,583,466] 
non1 = [932,292,233]
nor1 = [932,291,233]
cov2 =[7658,2395,1903] 
non2 = [7208,2253,1802]
nor2 = [6849,2140,1712]
data = np.array([cov1, non1, nor1])
data2 = np.array([cov2, non2, nor2])

bar_width = 0.2
bar_positions = np.arange(len(sets))
plotted_data = plt.Figure()
ax = plotted_data.add_subplot(111)

if dataset == "infection segmentation data":
		ax.bar(bar_positions, data[0], bar_width, color='blue', label='Covid-19', edgecolor='black')
		ax.bar(bar_positions + bar_width, data[1], bar_width, color='orange',label='Non-Covid', edgecolor='black')
		ax.bar(bar_positions + 2 * bar_width, data[2], bar_width, color='green', label='Normal', edgecolor='black')
elif dataset == "lung segmentation data":
		ax.bar(bar_positions, data2[0], bar_width, color='blue', label='Covid-19',edgecolor='black')
		ax.bar(bar_positions + bar_width, data2[1], bar_width, color='orange',label='Non-Covid', edgecolor='black')
		ax.bar(bar_positions + 2 * bar_width, data2[2], bar_width, color='green',label='Normal', edgecolor='black')
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(sets)
ax.set_xlabel('sets')
ax.set_ylabel('count')
ax.set_title('data balance')	
ax.legend()

buf = BytesIO()
plotted_data.savefig(buf, format="png", bbox_inches = "tight")
st.image(buf, width=450, use_column_width=False)

################################################################################

st.header("Creation of processed data sets")
st.markdown("We applied histogram equalization to enhance the contrast, making image information more apparent and learning easier for the model."   
" We applied and cropped the lung mask to focus on the lung region, where the desease is to be found.")
st.markdown('**Pixel intensity distribution and average images for each class of the dataset.**')
visual_pic = st.selectbox("Choose dataset",('unprocessed', 'equalized', 'cropped masks'))
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','Visualisation','{}_dist.png'.format(visual_pic))), "", 600, 600)
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','Visualisation','{}_mean.png'.format(visual_pic))), visual_pic, 550, 550)

##################################################################################

st.header("Augmentation")
width = st.sidebar.slider("zoom augmentation", 400, 1100, 700)
st.markdown('Data augmentation helps to enhance the variety of the data, improve the robustness of the model and can help preventing unwanted biases.')
aug_pic = st.selectbox("Choose transformation",('rotation_20', 'vertical flip', 
'horizontal flip', 'width shift_0.2', 'height shift_0.2', 'zoom_0.2', 'gaussian filter_1.2', 'gamma_0.2', 'contrast_0.6', 'brightness_0.5', 'all'))
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','Augmentation','normal.png')), "normal", width=width)
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','Augmentation','{}.png'.format(aug_pic))), aug_pic, width=width)

###################################################################################

st.header("Showing unwanted biases")
st.markdown("Simple CNN Model architecture.")
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models','simple1.png')),"" ,600, 600)
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models','simple2.png')),"",600, 600)             
st.markdown("Images with resolution (16*16).")
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models','example.png')),"",400, 400) 
st.markdown('**Results: Confusionmatrix, metrics and activation maps.**')
met_pic = st.selectbox("Choose dataset",('unprocessed', 'equalized', 'eq augmented'))
img1=Image.open(os.path.join(root,'data','#2. Data processing','imgs','metrics_confusion','{}_conf.png'.format(met_pic)))
img2=Image.open(os.path.join(root,'data','#2. Data processing','imgs','metrics_confusion','{}_met.png'.format(met_pic)))
img3=Image.open(os.path.join(root,'data','#2. Data processing','imgs','heatmaps','{}_hea.png'.format(met_pic)))

img1=img1.resize((300,300))
img2=img2.resize((500,400))
st.image([img1,img2], width=300)

if met_pic == 'eq augmented':
	img4=Image.open(os.path.join(root,'data','#2. Data processing','imgs','metrics_confusion','confbalance.png'))
	st.image(img4,"" ,300, 300)
width2 = st.sidebar.slider("bias zoom activation maps", 200, 700, 400)
st.image(img3,"" ,width= width2)

##################################################################################

st.header("First model approach on cropped lung masks")
st.markdown("Simple CNN Model 2 architecture.")
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models2','simple1.png')),"" ,600, 600)
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models2','simple2.png')),"",600, 600)
st.markdown("augmented trainset with resolution (180*180).")            
st.image(Image.open(os.path.join(root,'data','#2. Data processing','imgs','models2','example2.png')),"",500, 500)
st.markdown('**Results: Confusionmatrix, metrics and activation maps.**')
met_pic2 = st.selectbox("Choose segmentation",('infection segmentation', 'lung segmentation'))
img1=Image.open(os.path.join(root,'data','#2. Data processing','imgs','metrics_confusion2','{}_conf.png'.format(met_pic2)))
img2=Image.open(os.path.join(root,'data','#2. Data processing','imgs','metrics_confusion2','{}_met.png'.format(met_pic2)))
img3=Image.open(os.path.join(root,'data','#2. Data processing','imgs','heatmaps2','{}_hea.png'.format(met_pic2)))

img1=img1.resize((300,300))
img2=img2.resize((500,400))
st.image([img1,img2], width=300)
width3 = st.sidebar.slider("first model zoom activation maps", 400, 1000, 600)
st.image(img3,"" ,width=width3)



