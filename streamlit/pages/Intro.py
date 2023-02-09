# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:13:25 2023

@author: PC
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
#__________________________________ SIDEBAR ________________________________________
st.sidebar.markdown("# Introduction")


#__________________________________ PAGE CONTENT ________________________________________
print(os.getcwd())
root = os.path.dirname(os.path.realpath(__file__))
#print(root) 
st.markdown("# Covid-19 chest x-rays")
st.markdown("## Introduction")
st.markdown("- Covid-19 was discovered in the majority of nations by early spring 2020")
st.markdown("- Its symptoms are difficult to be identified from those of other viral infections ")
st.markdown("- RT-PCR is the gold standard diagnostic test")
st.markdown("- Several studies have suggested using chest computed tomography and Chest X-ray")
st.markdown("- Machine learning methods and Deep Learning techniques can improve efficiency and speed up the time to diagnose")


st.markdown("## Data set")
st.markdown("- A team of research with medical doctors from:")
st.text("- Qatar University\n" 
        "- University of Dhaka\n"
        "- Collaborators from Pakistan and Malaysia")
st.markdown("- A database of chest X-ray images for:")
st.text ("- COVID-19 positive cases (11,954)\n"
        "- Other lung infection (11,263)\n" 
        "- Normal cases (10,701)")



index= ['Covid-19','Non-COVID','Normal']  
train_ = [7658,7208,6849]  
test_ = [2394,2253,2140]
val_ = [1902,1802,1712]
 
chart_data = pd.DataFrame({"Train": train_, "Test": test_, "Validation": val_}, index=index)
    

st.bar_chart(chart_data)
#\

st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("## Data Exploration ")
st.markdown("#### Histogram Equalisation ")
st.markdown("We applied the histogram equalisation to improve the contrast of the images. Its function distributes effectively the apparent contrast perceived in an image")

image= Image.open(os.path.join(root+ r'/pages/data/Introduction/oneXray.png'))
         # 700, 600)
st.image(image)#, caption='Sunrise by the mountains')

st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### Outliers Detection ")
st.markdown("The unsupervised algorithm Isolation Forest has been used to detect the outliers in the data set. It recognised 85 images as outliers in the data set, however they haven’t been discarded from the database ")
image= Image.open(os.path.join(root+ r'/pages/data/Introduction/outliers.png'))
         # 700, 600)
st.image(image)#, caption='Sunrise by the mountains')

st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### Feature Correlation and Distribution")
st.markdown("We measured the linear association between variables to minimise the size of the feature space to an optimum number of features")
model_pic = st.selectbox("View Feature Correlation and Distribution",('Correlation', 'Distribution_of_Mean', 'Distribution_of_std', 'Distribution_of_skew', 'Distribution_of_kurt', 'Distribution_of_m00', 'Distribution_of_x_center',
                          'Distribution_of_y_center','Distribution_of_nu20','Distribution_of_nu11','Distribution_of_nu02','Distribution_of_nu30','Distribution_of_nu21','Distribution_of_nu12','Distribution_of_nu03'))

st.image(Image.open(os.path.join(root+'/pages/data/Introduction/{}.png'.format(model_pic))),
         model_pic, 700, 600)


st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### Data Variation")
st.markdown("To determine and explaine the variation in the data, we measured the eigenvectors of the Principal component analysis (PCA) to explains as much as possible of the total variation in the data set with the fewest possible PC")

image= Image.open(os.path.join(root+ r'/pages/data/Introduction/EigenVector.png'))
         # 700, 600)
st.image(image, caption='Principle Components')


st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### K-means clustering")
st.markdown("To evaluate the performance of the K-Means algorithm and to determine the optimal number of cluster in the data, the Sum of Squared Errors (SSE) in K-Means clustering is used")
image= Image.open(os.path.join(root+ r'/pages/data/Introduction/SSE.png'))
         # 700, 600)
st.image(image)#, caption='Principle Components')

st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### Clustering")
st.markdown("Using 10 clusters, We analyse the distribution of the classes using the first 2 principle components;")
st.markdown("- Diseases: It shows that no classes are really discriminate for isolating one of our target label")
st.markdown("- Data sets: It shows that the probability distribution of the clusters is a little bit dissimilar between the dataset")

model_pic = st.selectbox("The distribution of the classes",('Diseases_Distributions', 'Data_Distributions'))

st.image(Image.open(os.path.join(root+'/pages/data/Introduction/{}.png'.format(model_pic))),
         model_pic, 700, 600)

st.text("\n"
         "\n"
         "\n"
         "\n") 
st.markdown("#### The dissimilarities between the data sets")
st.markdown("They may lead to difficulties in obtaining good generalization results")

model_pic = st.selectbox("subset of images",('label_0', 'label_1','label_2', 'label_3','label_4', 'label_5','label_6', 'label_7','label_8', 'label_9'))

st.image(Image.open(os.path.join(root+'/pages/data/Introduction/{}.png'.format(model_pic))),
         model_pic, 700, 600)