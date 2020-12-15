import os
import glob
import time
from multiprocessing import Process

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from dataset_utils import preprocess_image
from model_v1_utils import load_model
from viz_utils import create_gradcam_image, create_gradcam_models, make_gradcam_heatmap
from utils import initialize_gpu, tf_resize_image, load_config, convert_to_pil_image_with_title,\
    fast_make_grid, fast_save_grid


def run_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()


def load_image_without_resizing(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image


def prepare_test_images(p, backbone, target_size):
    image = load_image_without_resizing(p)
    test_image = preprocess_image(image, backbone, training=False, input_preprocessing=False)
    test_input_image = preprocess_image(
        tf_resize_image(image, target_size), backbone, training=False, input_preprocessing=True
    )
    return test_image, test_input_image


def prepare_test_images_lists(paths, backbone, target_size):
    test_images_list = []
    test_input_images_list = []
    for p in paths:
        test_image, test_input_image = prepare_test_images(p, backbone, target_size)
        test_images_list.append(test_image.numpy())
        test_input_images_list.append(test_input_image.numpy())
    return test_images_list, test_input_images_list


def process_preds(preds):
    if preds.shape[1] == 1:
        preds = [round(float(p[0]), 3) for p in preds]
    elif preds.shape[1] == 2:
        preds = [round(float(p[1]), 3) for p in preds]
    else:
        preds = [list(p) for p in preds]
    return preds


def prepare_model_validation_image(images_paths, model_path, model=None, grid_target_size=None,
                                   out_dir=None, init_gpu=True):
    if init_gpu:
        initialize_gpu(mode='growth')

    if grid_target_size is None:
        grid_target_size = (256, 256)

    # Extract model information
    if model is None:
        model = load_model(model_path)

    model_name = os.path.split(model_path)[1]

    config = load_config(model_path)

    target_size = config['image_target_size']
    normalize_embeddings = config['normalize_embeddings']

    backbone = config['backbone']
    pooling = config['pooling']
    n_outputs = config['n_outputs']

    # Extract information for heatmap models
    base_model = model.layers[0]
    top_model = model.layers[1]

    last_conv_layer_model, classifier_model = create_gradcam_models(
        base_model, top_model,
        backbone=backbone,
        pooling=pooling,
        normalize_embeddings=normalize_embeddings
    )

    def image_heatmap(image):
        class_index = n_outputs - 1
        heatmap = make_gradcam_heatmap(
            tf.expand_dims(image, axis=[0]),
            last_conv_layer_model=last_conv_layer_model,
            classifier_model=classifier_model,
            class_index=class_index
        ).astype(np.float32)
        return heatmap

    # Loading of two lists of images: one for processing and the other one for plotting
    test_images_list, test_input_images_list = prepare_test_images_lists(images_paths, backbone, target_size)
    test_images_list = [
        tf.cast(tf_resize_image(image, grid_target_size), tf.uint8).numpy()
        for image in test_images_list
    ]

    # Processing of predictions
    preds = model(np.array(test_input_images_list)).numpy()
    preds = process_preds(preds)

    # Images titles
    def get_fname(p):
        return os.path.split(p)[1].split('.')[0]

    titles = [f'{get_fname(p)}: score={score}' for p, score in zip(images_paths, preds)]

    # Processing of heatmaps
    heatmaps = [image_heatmap(image) for image in test_input_images_list]
    heatmap_images = [
        np.asarray(create_gradcam_image(image / 255., heatmap))
        for image, heatmap in zip(test_images_list, heatmaps)
    ]

    # Title for each image
    title_height = 70
    title_font_size = 35
    ncols = 8
    nrows = None
    images = np.array([
        np.asarray(convert_to_pil_image_with_title(image, title))
        for image, title in zip(heatmap_images, titles)
    ])
    images_grid = fast_make_grid(images, ncols=ncols, nrows=nrows)
    images_grid = convert_to_pil_image_with_title(
        images_grid, model_name, title_height=title_height, title_font_size=title_font_size
    )

    # Saving
    if out_dir is not None:
        fast_save_grid(
            out_dir, fname=model_name, images=images, nrows=nrows, ncols=ncols,
            title_height=title_height, title_font_size=title_font_size, title=model_name
        )

    # For Jupyter notebooks
    return images_grid


def get_valid_images_paths():
    paths_1 = glob.glob('test_images/AudreyHepburn/*')
    paths_2 = glob.glob('test_images/JenniferAniston/*')
    paths_3 = glob.glob('test_images/Other/*')

    valid_images_paths = paths_1 + paths_2 + paths_3
    return valid_images_paths


def check_if_configs_are_available(models_paths):
    missing_configs = []
    for model_path in models_paths:
        try:
            load_config(model_path)
        except:
            missing_configs.append(model_path)

    if len(missing_configs) == 0:
        print('All configs are available!')
        state = True
    else:
        state = False
        print(f'\n\nConfigs not defined for {len(missing_configs)} models. Here they are:')
        for p in missing_configs:
            print(p)

    return state


if __name__ == '__main__':
    start_time = time.time()

    valid_images_paths = get_valid_images_paths()
    print(f'Total number of images: {len(valid_images_paths)}')

    out_dir = 'visualization_valid_images'
    grid_target_size = (256, 256)

    models_paths = glob.glob(os.path.join('models', 'v1', 'full', '*'))
    if check_if_configs_are_available(models_paths):
        for model_path in tqdm(models_paths):
            run_process(
                target=prepare_model_validation_image,
                args=(valid_images_paths, model_path, None, grid_target_size, out_dir, True)
            )
    else:
        print(
            'Add all missing configs to perform validation or just remove models without configs from '
            'models paths'
        )

    total_time = time.time() - start_time
    print(f'Validation took: {total_time:.3f} seconds')
