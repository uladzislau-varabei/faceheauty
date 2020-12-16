import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm_notebook, tqdm
import tensorflow as tf

from face_models.facenet.inception_resnet_v1 import InceptionResNetV1
from face_models.face_evoLVe_ir50.ir50 import IR50
from face_models.insightface.lresnet100e_ir import LResNet100E_IR
from utils import load_config


DROPOUT_RATE = 0.3
FACE_BACKBONES = ['facenet', 'face_evoLVe_ir50'.lower(), 'insightface']


# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb#scrollTo=4BpQqKIeglKl

def get_base_model(backbone, pooling=None, input_shape=None, normalize_embeddings=False, load_weights=True):
    base_model_kwargs = {
        'include_top': False, 'pooling': pooling, 'input_shape': input_shape,
        'weights': 'imagenet' if load_weights else None
    }
    lw_backbone = backbone.lower()
    if lw_backbone == 'mobilenet':
        base_model = tf.keras.applications.MobileNet(**base_model_kwargs)
    elif lw_backbone == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(**base_model_kwargs)
    elif lw_backbone == 'resnet50':
        base_model = tf.keras.applications.resnet.ResNet50(**base_model_kwargs)
    elif lw_backbone == 'resnet50v2':
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(**base_model_kwargs)
    elif lw_backbone == 'densenet121':
        base_model = tf.keras.applications.densenet.DenseNet121(**base_model_kwargs)
    elif lw_backbone == 'effnetB0'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(**base_model_kwargs)
    elif lw_backbone == 'effnetB1'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB1(**base_model_kwargs)
    elif lw_backbone == 'effnetB2'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB2(**base_model_kwargs)
    elif lw_backbone == 'effnetB3'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB3(**base_model_kwargs)
    elif lw_backbone == 'effnetB4'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB4(**base_model_kwargs)
    elif lw_backbone == 'effnetB5'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB5(**base_model_kwargs)
    elif lw_backbone == 'effnetB6'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB6(**base_model_kwargs)
    elif lw_backbone == 'effnetB7'.lower():
        base_model = tf.keras.applications.efficientnet.EfficientNetB7(**base_model_kwargs)
    elif lw_backbone == 'facenet':
        if load_weights:
            weights_path = os.path.join('face_models', 'facenet', 'weights', 'facenet_keras_weights.h5')
        else:
            weights_path = None
        add_kwargs = {'normalize_embeddings': normalize_embeddings, 'weights_path': weights_path}
        base_model = InceptionResNetV1(**{**base_model_kwargs, **add_kwargs})
    elif lw_backbone == 'face_evoLVe_ir50'.lower():
        if load_weights:
            weights_path = os.path.join('face_models', 'face_evoLVe_ir50', 'backbone_ir50_ms1m_keras.h5')
        else:
            weights_path = None
        add_kwargs = {'normalize_embeddings': normalize_embeddings, 'weights_path': weights_path}
        base_model = IR50(**{**base_model_kwargs, **add_kwargs})
    elif lw_backbone == 'insightface':
        if load_weights:
            weights_path = os.path.join('face_models', 'insightface', 'lresnet100e_ir_keras.h5')
        else:
            weights_path = None
        add_kwargs = {'normalize_embeddings': normalize_embeddings, 'weights_path': weights_path}
        base_model = LResNet100E_IR(**{**base_model_kwargs, **add_kwargs})
    else:
        assert False, 'Unknown backbone'

    return base_model


def create_top_model(n_outputs, n_layers, units, act, use_batchnorm, use_dropout,
                     dropout_rate=DROPOUT_RATE, input_shape=None):
    def create_block(idx):
        act_dict = {
            'leaky_relu': tf.keras.layers.LeakyReLU()
        }
        layers = [
            tf.keras.layers.Dense(units, kernel_initializer=block_dense_init)
        ]
        if use_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        if act in act_dict.keys():
            layers.append(act_dict[act])
        else:
            layers.append(tf.keras.layers.Activation(act))
        if use_dropout:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        return tf.keras.models.Sequential(layers, name=f'Block_{idx+1}')

    if act.lower() != 'selu':
        block_dense_init = tf.keras.initializers.HeUniform()
    else:
        block_dense_init = tf.keras.initializers.LecunUniform()

    blocks = []
    if input_shape is not None:
        blocks.append(tf.keras.layers.Input(shape=input_shape, name='Inputs'))
    for idx in range(n_layers):
        blocks.append(create_block(idx=idx))

    output_act = tf.nn.sigmoid if n_outputs == 1 else tf.nn.softmax
    blocks.append(tf.keras.layers.Dense(n_outputs))
    blocks.append(tf.keras.layers.Activation(output_act, name='Outputs', dtype=tf.float32))

    return tf.keras.models.Sequential(blocks, name='Top_model')


def create_model(input_shape, backbone, normalize_embeddings, pooling, n_outputs, n_layers, units, act,
                 use_batchnorm, use_dropout, dropout_rate=DROPOUT_RATE):
    base_model = get_base_model(
        backbone, pooling, input_shape=input_shape, normalize_embeddings=normalize_embeddings
    )
    top_model = create_top_model(
        n_outputs, n_layers, units, act, use_batchnorm, use_dropout, dropout_rate
    )
    return tf.keras.Sequential([base_model, top_model])


def create_bottleneck_features(base_model, dataset, dataset_size, dtype, jupyter_mode=False):
    bottleneck_iter = iter(dataset)
    bottleneck_start_idx = 0
    bottleneck_preds = np.zeros((dataset_size, base_model.output_shape[1]), dtype=dtype)

    iter_flag = True
    tqdm_pbar = tqdm_notebook(desc='Batches') if jupyter_mode else tqdm(desc='Batches')
    with tqdm_pbar as pbar:
        while iter_flag:
            try:
                batch_images = next(bottleneck_iter)
                batch_preds = base_model(batch_images, training=False)
                for idx, bottleneck_idx in enumerate(
                        range(bottleneck_start_idx, bottleneck_start_idx + len(batch_preds))
                ):
                    bottleneck_preds[bottleneck_idx] = batch_preds[idx]
                bottleneck_start_idx += len(batch_preds)
                pbar.update(1)
            except (tf.errors.OutOfRangeError, StopIteration):
                print('Dataset exhausted')
                iter_flag = False

    return bottleneck_preds


def convert_face_model_to_fp32(model, config):
    print('\nConverting to fp32 model...')

    # Interesting if session clearing helps to use several models within one script
    # Note: without this line there might be some problems
    tf.keras.backend.clear_session()
    weights = deepcopy(model.get_weights())

    target_size = config['image_target_size']
    input_shape = target_size + [3]
    normalize_embeddings = config['normalize_embeddings']

    backbone = config['backbone']
    pooling = config['pooling']
    n_outputs = config['n_outputs']
    n_layers = config['layers']
    units = config['units']
    act = config['act']
    use_batchnorm = config['use_batchnorm']
    use_dropout = config['use_dropout']
    dropout_rate = config['dropout_rate']

    base_model = get_base_model(
        backbone, pooling, input_shape, normalize_embeddings=normalize_embeddings, load_weights=False
    )
    base_model.trainable = False
    top_model = create_top_model(
        n_outputs=n_outputs,
        n_layers=n_layers,
        units=units,
        act=act,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        input_shape=(base_model.output_shape[1],)
    )

    converted_full_model = tf.keras.Sequential([base_model, top_model])
    converted_full_model.set_weights(weights)

    return converted_full_model


def load_model(model_path, should_convert_face_model_to_fp32=True):
    # Running models in mixed precision mode on cpu is extremely slow
    # First try loading config as model loading takes far more time
    config = load_config(model_path)
    backbone = config['backbone'].lower()

    model = tf.keras.models.load_model(model_path, compile=False)

    if should_convert_face_model_to_fp32:
        if backbone in FACE_BACKBONES:
            model = convert_face_model_to_fp32(model, config)

    return model
