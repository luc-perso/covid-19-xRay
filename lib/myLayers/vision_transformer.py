import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from myLayers.mlp import mlp


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        # print(tf.shape(images))
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# todo before call
# inputs = layers.Input(shape=input_shape)
# # Augment data.
# augmented = data_augmentation(inputs)
# estimate image_size from augmented shape
def add_vit(input_layer,
            patch_size=8,
            input_image_size=256,
            transformer_layers=4,
            num_heads=4,
            projection_dim=64,
            transformer_units_rate=[2, 1],
            mlp_head_units=[2048, 1024]):
    transformer_units = [rate * projection_dim for rate in transformer_units_rate]  # Size of the transformer layers

    # Create patches.
    patches = Patches(patch_size)(input_layer)
    # Encode patches.
    num_patches = (input_image_size // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    return features
# todo after call
# # Classify outputs.
# logits = layers.Dense(num_classes)(features)
# # Create the Keras model.
# model = keras.Model(inputs=inputs, outputs=logits)
# return model

