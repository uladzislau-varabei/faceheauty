import albumentations as A
import numpy as np
import tensorflow as tf

from utils import tf_resize_image


def load_image(path, target_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    # Preprocessing from the repo with data
    image = tf_resize_image(image, (256, 256))
    image = tf.image.central_crop(image, 224 / 256)
    # Possibly no-op
    image = tf_resize_image(image, target_size)
    image = tf.cast(tf.clip_by_value(image, 0, 255), dtype=tf.uint8)
    return image


def augment_image(image):
    # Works with single image
    transform = A.Compose([
        A.HorizontalFlip(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.IAASharpen(p=1.),
            A.IAAEmboss(p=1.),
            A.RandomBrightnessContrast(p=1.),
        ], p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=5,
            val_shift_limit=5,
            p=0.3
        ),
    ])

    return transform(image=image)['image']


def get_backbone_preprocessing(backbone):
    lw_backbone = backbone.lower()
    if 'mobilenet' in lw_backbone:
        if 'v2' in lw_backbone:
            input_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            input_fn = tf.keras.applications.mobilenet.preprocess_input
    elif 'resnet' in lw_backbone:
        if 'v2' in lw_backbone:
            input_fn = tf.keras.applications.resnet_v2.preprocess_input
        else:
            input_fn = tf.keras.applications.resnet.preprocess_input
    elif 'effnet' in lw_backbone:
        input_fn = tf.keras.applications.efficientnet.preprocess_input
    elif 'densenet' in lw_backbone:
        input_fn = tf.keras.applications.densenet.preprocess_input
    # Other networks pretrained on faces
    elif lw_backbone == 'facenet':
        # Confirmed here: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py (line 122)
        input_fn = tf.image.per_image_standardization
    elif lw_backbone == 'face_evoLVe_ir50'.lower():
        # Preprocessing: (x - 127.5) / 128
        input_fn = lambda x: (x - 127.5) / 128
    elif lw_backbone == 'insightface':
        # Preprocessing: (x - 127.5) / 128
        input_fn = lambda x: (x - 127.5) / 128
    else:
        assert False, 'Unknown backbone'

    return input_fn


def preprocess_image(image, backbone, training, input_preprocessing=True):
    if training:
        image = tf.numpy_function(func=augment_image, inp=[image], Tout=tf.uint8)
        image = tf.convert_to_tensor(image, tf.uint8)

    if input_preprocessing:
        image = tf.cast(image, dtype=tf.float32)
        input_fn = get_backbone_preprocessing(backbone)
        image = input_fn(image)

    return image


def get_image_score(p, target_variable_dict):
    if isinstance(p, bytes):
        p_key = p.decode('utf-8')
    else:
        # For compatibility with eager mode
        p_key = str(p)

    return np.array(target_variable_dict[p_key], dtype=np.float32)


def load_and_preprocess_image(path, target_size, backbone,
                              training, input_preprocessing, target_variable_dict):
    image = load_image(path, target_size)
    image = preprocess_image(image, backbone, training, input_preprocessing=input_preprocessing)
    # Score should always be tf.float32 (even in mixed precision mode)
    score = tf.numpy_function(func=lambda x: get_image_score(x, target_variable_dict), inp=[path], Tout=tf.float32)
    return image, score


def create_dataset(files, target_size, backbone, training, input_preprocessing, target_variable_dict,
                   batch_size):
    ds = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)
    ds = ds.map(lambda x: load_and_preprocess_image(
        x, target_size=target_size, backbone=backbone,
        training=training, input_preprocessing=input_preprocessing,
        target_variable_dict=target_variable_dict
    ))
    # Note: no need to repeat dataset when using mode.fit() method.
    # Number of steps for training/validation should not be specified
    ds = ds.batch(batch_size=batch_size, drop_remainder=training)
    prefetch_size = 4
    ds = ds.prefetch(buffer_size=prefetch_size)
    return ds
