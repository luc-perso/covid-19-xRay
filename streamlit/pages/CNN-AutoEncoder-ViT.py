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
st.markdown("""
## Introduction
<div style="text-align: justify;">
We try to build and learn a network with our own simple CNN pre-trained feature model.
The idea is to build a NN with 3 successive main blocks:

* CNN to reduce feature dimensionality
* Transformer to use information from different place in the image
* Fully connected layer for the classification
</div>
""",
unsafe_allow_html=True)

#---------------------------- BASE MODELS -----------------------------------------------
st.markdown("---")
st.markdown("## Model architectures")
st.markdown("### Sub-model architectures")
base_model = st.selectbox(
  "Choose a sub-model",
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

st.markdown("<br>", unsafe_allow_html=True)
#--------------------------------- COMPUND MODELS --------------------------------------
st.markdown("### Full model architectures")
compound_model = st.selectbox(
  "Choose a model",
  ('CNN-Autoencoder-Classifier', 'CNN-Classifier', 'CNN-ViT-Classifier')
)

match compound_model:
  case 'CNN-Autoencoder-Classifier':
    st.markdown("""
    |Layer (type)|Output Shape|Param #|Connected to|
    |------------|------------|-------|------------|
    |InputLayer|[(None, 256, 256, 1)]|0|[]|
    |Augmentation|(None, 256, 256, 1)|0|['inputLayer[0][0]']|
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
    |Augmentation|(None, 256, 256, 1)|0|['inputLayer[0][0]']|
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
    |Augmentation|(None, 256, 256, 1)|0|['inputLayer[0][0]']|
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
st.markdown("---")
st.markdown("""
## Learning path
<div style="text-align: justify;">
To let us a chance to make this kind of Neuronal Network works and learns, we build it by following these steps:

*	Pre-trained the CNN with an autoencoder structure to let it learn to compress the image in a feature space with fewer dimensions while keeping enough information to reconstruct the initial image.
*	Unplug the reconstruction part of the autoencoder scheme, and fine tune the CNN on the classification task.
*	Add a transformer network between the CNN and the classification task. Fix the CNN parameters and learn the transformer and classifier ones.
*	Fine tuning all the parameters of the model.

For all the different models, we use:
* An augmentation layer,
* Hot encoded labels,
* An AdamW optimizer, with:
  * A base learning rate: 		lr = 0.001,
  * A weight decay: 		w = 0.0001,
  * Other default parameter: 	β_1  = 0.999,β_2= 0.99
* Early stopping based on accuracy metrics,
* Learning on the Lung Segmentation Data images with histogram equalization and no use of lung or infection masks.
</div>
""",
unsafe_allow_html=True)

lr_curve = st.selectbox(
  "Choose a model",
  ('CNN-Autoencoder-Classifier', 'CNN-ViT-Classifier CNN locked', 'CNN-ViT-Classifier all layers')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'learning_path','{}.png'.format(lr_curve))),
  lr_curve,
  700, 600)

st.markdown("---")
st.markdown("""
<div style="text-align: justify;">
<strong>Note:</strong> we can observe on our learning curves that the validation accuracy (and the test one) is often above the train one. \
This is a sign that the train, valid and test dataset are not sampled from the same probability distribution. \
That overlaps our observations during data exploration.
</div>
""",
unsafe_allow_html=True)

#-------------------------------- Learning Results -----------------------------------
st.markdown("---")
st.markdown("## Comparative results")
st.markdown("**CNN-Classifier vs CNN-ViT-Classifier**")
st.markdown("""
    |class|precision|recall|f1-score|support|
    |-----|---------|------|--------|-------|
    |normal|0.91 / 0.94|0.93 / 0.89|0.92 / 0.92|2140|
    |covid|0.97 / 0.98|0.94 / 0.96|0.96 / 0.97|2394|
    |no-covid|0.91 / 0.88|0.93 / 0.95|0.92 / 0.91|2253|
    |accuracy| | |0.93|6787|
    |macro avg|0.93|0.93|0.93|6787|
    |weighted avg|0.93|0.93|0.93|6787|
""")
# st.markdown("## CNN-ViT-Classifier")
# st.markdown("""
#     |class|precision|recall|f1-score|support|
#     |-----|---------|------|--------|-------|
#     |normal|0.94|0.89|0.92|2140|
#     |covid|0.98|0.96|0.97|2394|
#     |no-covid|0.88|0.95|0.91|2253|
#     |accuracy| | |0.93|6787|
#     |macro avg|0.93|0.93|0.93|6787|
#     |weighted avg|0.93|0.93|0.93|6787|
# """)

st.markdown("---")
st.markdown("""
<div style="text-align: justify;">
The cnn-ViT-classifier model performance is equivalent to the cnn-classifier model and not really better. \
But it’s already an improvement to prove that such an architecture can learn to produce such a classification task. \
It will be interesting to test these both structures with a cnn composed with more layers and/or with a larger classifier.
</div>
""",
unsafe_allow_html=True)

#-------------------------------- Activation Map -----------------------------------
st.markdown("---")
st.markdown("""
## Visualization
To understand how our network works and when it is failing, we present some visualizations.

### Activation map
In Conv2D (None, 64, 64, 64) layer.
""")
activation = st.selectbox(
  "Choose an example",
  ('activation_1', 'activation_2')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'activation','{}.png'.format(activation))),
  activation,
  700, 600)

#-------------------------------- Pattern Map -----------------------------------
st.markdown("### CNN Pattern")
st.markdown("Pattern learned by convolution layers")
pattern = st.selectbox(
  "Choose a CNN layer",
  ('pattern_conv_layer_0', 'pattern_conv_layer_2', 'pattern_conv_layer_4', 'pattern_conv_layer_6')
)
st.image(Image.open(
  os.path.join(root, 'data', 'CNN-AutoEncoder-ViT', 'pattern','{}.png'.format(pattern))),
  pattern,
  700, 600)

st.markdown("---")
st.markdown("""
<div style="text-align: justify;">
We observe 2 types of patterns:

* Structural ones that detect:
  * Edges, corners in the first 2 layers,
  * Gabor style filter in the third and fourth one,
* Textured ones that seem noisy but detect specific texture. \
  This is useful to characterize different anatomical parts in the image as bones, lung, covid lung…
</div>
""",
unsafe_allow_html=True)
st.markdown("---")

#-------------------------------- Occultation vs Grad-CAM -----------------------------------
st.markdown("### Sensitivity map")

st.markdown("Occultation vs Grad-CAM.")
occ_vs_gc = st.selectbox(
  "Choose an example",
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

st.markdown("---")
st.markdown("""
<div style="text-align: justify;">
<strong>Note:</strong> The sensitivity map can present strong activation in the corner of the image that seems to lead the network answer. \
It’s often due to the presence of a letter or symbol on the x-ray. We should study if the presence of specific symbols is not unbalanced \
between the class to prevent classification algorithms to learn for instance than if C is written the patient has covid. \
If it is the case, this image could be discarded or tampered to erase such information.
</div>
""",
unsafe_allow_html=True)
st.markdown("---")

#-------------------------------- Grad-CAM -----------------------------------
st.markdown("### Grad-CAM on sub-groups")
grad_cam = st.selectbox(
  "Choose a groups.",
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

st.markdown("---")
st.markdown("""
<div style="text-align: justify;">
In all plots, the model pays attention to the right area i.e. the lungs, with no explicit signs that could explain the mistakes. \
A further analysis would be necessary to find the axis of improvement.

In some cases, the model focuses on the heart or shoulder joints too. \
It could be interesting to have doctors’ opinion on that point.
</div>
""",
unsafe_allow_html=True)
st.markdown("---")
