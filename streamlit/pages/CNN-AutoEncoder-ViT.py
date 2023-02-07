# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:15:36 2023

@author: PC
"""

import streamlit as st
import os
from PIL import Image
#__________________________________ SIDEBAR ________________________________________
st.sidebar.markdown("# CNN-Autoencoder-ViT")

#__________________________________ PAGE CONTENT ________________________________________
root = os.path.dirname(os.path.realpath(__file__))

st.markdown("# CNN-Autoencoder-ViT")
st.markdown("We try to build and learn a network with our own simple CNN pre-trained feature model.")

#---------------------------- BASE MODELS -----------------------------------------------
base_model = st.selectbox(
  "View base model architectures",
  ('Augmentation', 'CNN-Encoder', 'Fully-Connected Decoder', 'Classifier', 'ViT')
)

match base_model:
  case 'Augmentation':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|
    |------------|------------|-------|
    |rescaling (Rescaling)|(None, 256, 256, 1)|0|
    |random_flip (RandomFlip)|(None, 256, 256, 1)|0|
    |random_rotation (RandomRotation)|(None, 256, 256, 1)|0|
    |random_zoom (RandomZoom)|(None, 256, 256, 1)|0|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|0|
    |Trainable params|0|
    |Non-trainable params|0|
    """)
  case 'CNN-Encoder':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|
    |------------|------------|-------|
    |conv2d (Conv2D)|(None, 256, 256, 128)|1280|
    |max_pooling2d (MaxPooling2D)|(None, 128, 128, 128)|0|
    |conv2d_1 (Conv2D) |(None, 128, 128, 128)|147584|
    |max_pooling2d_1 (MaxPooling2D)|(None, 64, 64, 128)|0|
    |conv2d_2 (Conv2D)|(None, 64, 64, 64)|73792|
    |max_pooling2d_2 (MaxPooling2D)|(None, 32, 32, 64)|0|
    |conv2d_3 (Conv2D)|(None, 32, 32, 64)|36928|
    |max_pooling2d_3 (MaxPooling2D)|(None, 16, 16, 64)|0|
    |conv2d_4 (Conv2D)|(None, 16, 16, 64)|36928|
    |flatten (Flatten)|(None, 4096)|0|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|296,512|
    |Trainable params|296,512|
    |Non-trainable params|0|
    """)
  case 'Fully-Connected Decoder':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|
    |------------|------------|-------|
    |dense_2 (Dense)|(None, 1024)|4195328|
    |dense_3 (Dense)|(None, 256)|262400|
    |dense_4 (Dense)|(None, 65536)|16842752|
    |reshape (Reshape)|(None, 256, 256, 1)|0|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|21,300,480|
    |Trainable params|21,300,480|
    |Non-trainable params|0|
    """)
  case 'Classifier':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|
    |------------|------------|-------|
    |dense_5 (Dense)|(None, 1024)|4195328|
    |dense_6 (Dense)|(None, 256)|262400|
    |dense_7 (Dense)|(None, 3)|771|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|4,458,499|
    |Trainable params|4,458,499|
    |Non-trainable params|0|
    """)
  case 'ViT':
    st.markdown("""
    |Layer (type)|Output_Shape|Param_#|Connected to|
    |------------|------------|-------|------------|
    |input_1 (InputLayer)|[(None, 8, 8, 64)]|132672|[]|
    |patches|(None, 64, 64)|0|['input_1[0][0]']|
    |patch_encoder|(None, 64, 64)|8256|['patches[0][0]']|
    |layer_normalization|(None, 64, 64)|128|['patch_encoder[0][0]']|
    |Block to repeat N times||||||
    |multi_head_attention|(None, 64, 64)|132672|['layer_normalization[0][0]', 'layer_normalization[0][0]']|
    |add (Add) |(None, 64, 64)|0|['multi_head_attention[0][0]', 'patch_encoder[0][0]']|
    |layer_normalization_1|(None, 64, 64)|128|['add[0][0]']|
    |dense_7 (Dense)|(None, 64, 128)|8320|[['layer_normalization_1[0][0]']]|
    |dropout (Dropout)|(None, 64, 128)|0|['dense_7[0][0]']|
    |dense_8 (Dense)|(None, 64, 64)|8256|['dropout[0][0]']|
    |dropout_1 (Dropout)|(None, 64, 64)|0|['dense_8[0][0]']|
    |add_1 (Add)|(None, 64, 64)|0|['dropout_1[0][0]', 'add[0][0]'] |
    |layer_normalization_2|(None, 64, 64)|128|['add_1[0][0]']|


    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|8,384 + N * 149,504|
    |Trainable params|8,384 + N * 149,504|
    |Non-trainable params|0|
    """)
  case _:
    pass

#--------------------------------- COMPUND MODELS --------------------------------------
st.markdown("\n\n")
compound_model = st.selectbox(
  "View compound model architectures",
  ('CNN-Autoencoder-Classifier', 'CNN-Classifier', 'CNN-ViT-Classifier')
)

match compound_model:
  case 'CNN-Autoencoder-Classifier':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|Connected to|
    |------------|------------|-------|------------|
    |InputLayer|[(None, 256, 256, 1)]|0|[]|
    |Augmentation|(None, 256, 256, 1)|0|['input_1[0][0]']|
    |Encoder|(None, 4096)|296512|['augmentation[0][0]']|
    |Decoder|(None, 256, 256, 1)|21300480|['encoder[0][0]']|
    |Classifier|(None, 3)|4458499|['encoder[0][0]']|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|26,055,491|
    |Trainable params|26,055,491|
    |Non-trainable params|0|
    """)
  case 'CNN-Classifier':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|Connected to|
    |------------|------------|-------|------------|
    |InputLayer|[(None, 256, 256, 1)]|0|[]|
    |Augmentation|(None, 256, 256, 1)|0|['input_1[0][0]']|
    |Encoder|(None, 4096)|296512|['augmentation[0][0]']|
    |Classifier|(None, 3)|4458499|['encoder[0][0]']|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|4,755,011|
    |Trainable params|4,755,011|
    |Non-trainable params|0|
    """)
  case 'CNN-ViT-Classifier':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|Connected to|
    |------------|------------|-------|------------|
    |InputLayer|[(None, 256, 256, 1)]|0|[]|
    |Augmentation|(None, 256, 256, 1)|0|['input_1[0][0]']|
    |Encoder|(None, 4096)|296512|['augmentation[0][0]']|
    |ViT (4 multihead attention layers)|(None, 4096)|606400|['encoder[0][0]']|
    |Classifier|(None, 3)|4458499|['ViT[0][0]']|
    """)
    st.markdown("""
    ||Param #|
    |------------|------------|
    |Total params|5,361,411|
    |Trainable params|5,361,411|
    |Non-trainable params|0|
    """)
  case _:
    pass

#-------------------------------- Learning Curves -----------------------------------
lr_curve = st.selectbox(
  "Learning Path",
  ('CNN-Autoencoder-Classifier', 'CNN-ViT-Classifier CNN locked', 'CNN-ViT-Classifier all layers')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'learning_path','{}.png'.format(lr_curve))),
  lr_curve,
  700, 600)

#-------------------------------- Learning Results -----------------------------------
result = st.selectbox(
  "Results",
  ('CNN-Classifier', 'CNN-ViT-Classifier')
)
match result:
  case 'CNN-Classifier':
    st.markdown("""
    |class|precision|recall|f1-score|support|
    |-----|---------|------|--------|-------|
    |normal|0.91|0.93|0.92|2140|
    |covid|0.97|0.94|0.96|2394|
    |no-covid|0.91|0.93|0.92|2253|
    |accuracy| | |0.93|6787|
    |macro avg|0.93|0.93|0.93|6787|
    |weighted avg|0.93|0.93|0.93|6787|
    """)
  case 'CNN-ViT-Classifier':
    st.markdown("""
    |class|precision|recall|f1-score|support|
    |-----|---------|------|--------|-------|
    |normal|0.94|0.89|0.92|2140|
    |covid|0.98|0.96|0.97|2394|
    |no-covid|0.88|0.95|0.91|2253|
    |accuracy| | |0.93|6787|
    |macro avg|0.93|0.93|0.93|6787|
    |weighted avg|0.93|0.93|0.93|6787|
    """)
  case _:
    pass

#-------------------------------- Activation Map -----------------------------------
activation = st.selectbox(
  "Example of activation map in conv2D (None, 64, 64, 64) layer.",
  ('activation_1', 'activation_2')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'activation','{}.png'.format(activation))),
  activation,
  700, 600)

#-------------------------------- Pattern Map -----------------------------------
pattern = st.selectbox(
  "Example of pattern detected by convolution layers.",
  ('pattern_conv_layer_0', 'pattern_conv_layer_2', 'pattern_conv_layer_4', 'pattern_conv_layer_6')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'pattern','{}.png'.format(pattern))),
  pattern,
  700, 600)

#-------------------------------- Occultation vs Grad-CAM -----------------------------------
occ_vs_gc = st.selectbox(
  "Occultation vs Grad-CAM.",
  ('1', '2', '3', '4', 'covid')
)
image_occultation = Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'occultation','occultation_{}.png'.format(occ_vs_gc)))
image_gc = Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'grad_cam','grad_cam_{}.png'.format(occ_vs_gc)))
st.image(
  [image_occultation, image_gc],
  [occ_vs_gc, occ_vs_gc],
  350, 350)

#-------------------------------- Grad-CAM -----------------------------------
grad_cam = st.selectbox(
  "Grad-CAM of specific group.",
  ('normal', 'covid', 'no-covid',
  'covid as normal', 'false_covid_detection', 'normal_miss_classified',
  'false_detection', 'true_detection')
)
image_grad_cam = Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'grad_cam','{}.png'.format(grad_cam)))
st.image(
  image_grad_cam,
  grad_cam,
  700, 600)
