import numpy as np
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf

from model_v1_utils import FACE_BACKBONES
from utils import l2_normalizer, DEFAULT_CMAP


### ---------- Guided Grad-CAM ----------

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


### ---------- Grad-CAM ----------

def create_gradcam_models(base_model, top_model, backbone, pooling,
                          normalize_embeddings=True, last_conv_layer_dict=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    if last_conv_layer_dict is None:
        last_conv_layer_dict = {
            'resnet50v2': 'post_relu',
            'resnet50': 'conv5_block3_out',
            'mobilenetv2': 'out_relu',
            'mobilenet': 'conv_pw_13_relu',
            'densenet121': 'relu',
            'effnetB0': 'top_activation',
            'effnetB1': 'top_activation',
            'effnetB2': 'top_activation',
            'effnetB3': 'top_activation',
            'effnetB4': 'top_activation',
            'effnetB5': 'top_activation',
            'effnetB6': 'top_activation',
            'effnetB7': 'top_activation',
            # 'facenet': 'Block8_6_ScaleSum',
            'facenet': 'add_20',
            'face_evoLVe_ir50'.lower(): 'dropout',
            'insightface': 'dropout0'
        }
    lw_backbone = backbone.lower()
    last_conv_layer_name = last_conv_layer_dict[lw_backbone]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.layers.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    if lw_backbone in FACE_BACKBONES:
        if lw_backbone == 'facenet':
            layers = [
                base_model.get_layer('AvgPool'),
                base_model.get_layer('Dropout'),
                base_model.get_layer('Bottleneck'),
                base_model.get_layer('Bottleneck_BatchNorm')
            ]
        elif lw_backbone == 'face_evoLVe_ir50'.lower():
            layers = [
                base_model.get_layer('permute'),
                base_model.get_layer('flatten'),
                base_model.get_layer('output_layer.3'),
                base_model.get_layer('output_layer.4')
            ]
        elif lw_backbone == 'insightface':
            layers = [
                base_model.get_layer('permute'),
                base_model.get_layer('flatten'),
                base_model.get_layer('pre_fc1'),
                base_model.get_layer('fc1')
            ]
        else:
            layers = []
        if normalize_embeddings:
            # Layer might not be a part of the model
            layers.append(l2_normalizer())
        for layer in layers:
            x = layer(x)
    else:
        assert pooling in ['avg', 'max']
        if pooling == 'avg':
            pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        else:
            pool_layer = tf.keras.layers.GlobalMaxPooling2D()
        x = pool_layer(x)
    x = top_model(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    return last_conv_layer_model, classifier_model


def make_gradcam_heatmap(image, last_conv_layer_model, classifier_model, class_index=None):
    # Create last conv layer and classifier models (for efficiency previous steps)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = tf.cast(last_conv_layer_model(image), tf.float32)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = tf.cast(classifier_model(last_conv_layer_output), tf.float32)
        if class_index is not None:
            top_pred_index = class_index
        else:
            top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def create_gradcam_image(image, heatmap, mask_coords=None, smooth=False, cmap_name=DEFAULT_CMAP):
    heatmap_target_size = (image.shape[1], image.shape[0])
    if mask_coords is not None:
        mask_h_min, mask_h_max, mask_w_min, mask_w_max = mask_coords
        mask = np.zeros(shape=image.shape[:2], dtype=np.float32)
        if smooth:
            mask[:, :] = 0.15
        mask[mask_h_min: mask_h_max, mask_w_min: mask_w_max] = 1.

        heatmap_image = Image.fromarray(np.uint8(255 * heatmap), 'L')
        heatmap_image = heatmap_image.resize(heatmap_target_size, resample=Image.LANCZOS)

        heatmap = np.float32(np.array(heatmap_image) / 255.)

        heatmap *= mask
        # Renormalize heatmap (for cases when max value was outside of face)
        heatmap /= heatmap.max()
        # Heatmap is 2d array of np.float32 in range (0, 1)

    # We rescale images to a range 0-255
    rescaled_image = np.uint8(255 * image)
    rescaled_heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    cmap = cm.get_cmap(cmap_name)

    # We use RGB values of the colormap
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[rescaled_heatmap]

    # We create an image with RGB colorized heatmap
    cmap_heatmap = tf.keras.preprocessing.image.array_to_img(cmap_heatmap)
    # Only resize image if mask wasn't applied
    if mask_coords is None:
        cmap_heatmap = cmap_heatmap.resize(heatmap_target_size, resample=Image.LANCZOS)
    cmap_heatmap = tf.keras.preprocessing.image.img_to_array(cmap_heatmap)

    # Superimpose the heatmap on original image
    superimposed_image = cmap_heatmap * 0.3 + rescaled_image
    superimposed_image = tf.keras.preprocessing.image.array_to_img(superimposed_image)

    return superimposed_image


### ---------- Guided backpropagation ----------
# Thanks to https://stackoverflow.com/questions/55924331/how-to-apply-guided-backprop-in-tensorflow-2-0 (Hoa Nguyen)

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy
    return tf.nn.relu(x), grad


def convert_to_guided_backprop_model(model):
    gb_model = tf.keras.models.clone_model(model)
    act_layers = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]
    for layer in act_layers:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu
    return gb_model


def compute_gb_model_grads(image, gb_model):
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        tape.watch(inputs)
        outputs = gb_model(inputs)

    grads = tape.gradient(outputs, inputs)[0]
    return grads.numpy()


def deprocess_image(x, advanced=True, std=0.25):
    if advanced:
        """
        Same normalization as in:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        """
        # normalize tensor: center on 0., ensure std is 0.25
        x = x.copy()
        x -= x.mean()
        x /= (x.std() + tf.keras.backend.epsilon())
        x *= std

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
    else:
        x = x.copy()
        x -= x.min()
        x /= x.max()
        x = np.clip(x * 255, 0, 255).astype('uint8')
    return x
