import os
from io import BytesIO
import time

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

from inference import InferenceBeautyApp
from utils import initialize_gpu, VALUE_STABLE, VALUE_EXPERIMENTAL, VALUE_YES, VALUE_NO, \
    DEFAULT_APPLY_MASK, DEFAULT_CMAP


app = Flask(__name__)

STABLE_MODEL_PATH = 'models/v1/full/v1_full_facenet_normembeds_size300_stable_dataALL_out1_CE_adam0.001_sgd1e-05_decay0.0001_fzBN_final'
EXPERIMENTAL_MODEL_PATH = 'models/v1/full/v1_full_insightface_normembeds_size112_stable_dataALL_out1_CE_adam0.001_sgd1e-05_decay0.0001_trBN_epoch300'


# If set to True additional model will be loaded
LOAD_EXPERIMENTAL_MODEL = True


def create_inference_classes():
    start_time = time.time()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    initialize_gpu(mode='growth')
    inference_classes_dict = {}
    # Stable model
    inference_class_1 = InferenceBeautyApp(STABLE_MODEL_PATH)
    inference_classes_dict[VALUE_STABLE] = inference_class_1
    # Experimental model
    if LOAD_EXPERIMENTAL_MODEL:
        print('\nAlso loading experimental model...\n')
        inference_class_2 = InferenceBeautyApp(EXPERIMENTAL_MODEL_PATH)
        inference_classes_dict[VALUE_EXPERIMENTAL] = inference_class_2

    total_time = time.time() - start_time
    print(f'\nModels loaded in {total_time:.3f} seconds!\n')
    return inference_classes_dict


inference_classes_dict = create_inference_classes()


def bytes_to_array(data):
    to_rgb = lambda x: cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
    return to_rgb(Image.open(BytesIO(data)))


@app.route('/<model_type>/<apply_mask>/<palette>', methods=['GET', 'POST'])
def process_image_advanced(model_type, apply_mask, palette):
    start_time = time.time()

    image_bytes = request.get_data()
    image_array = bytes_to_array(image_bytes)

    if model_type in inference_classes_dict.keys():
        inference_class = inference_classes_dict[model_type]
    else:
        print(f'\nNo inference class for model_type={model_type}, using {VALUE_STABLE} instead')
        inference_class = inference_classes_dict[VALUE_STABLE]

    apply_mask = {
        VALUE_YES.lower(): True, VALUE_NO.lower(): False
    }.get(apply_mask.lower(), True)
    smooth = True

    output = inference_class.prepare_output(
        image_array, apply_mask=apply_mask, smooth=smooth, cmap_name=palette, output_json=True
    )

    total_time = time.time() - start_time
    print(f'Image processed in {total_time:.3f} seconds')

    return jsonify(output), 200


@app.route('/', methods=['GET', 'POST'])
def process_image_basic():
    # Raw binary response: flask.Response(bytes)
    # Note: json output is very slow on images where face has high resolution,
    # e.g. 1500x1500, up to 10 sec for 2 faces on 1 image

    start_time = time.time()

    image_bytes = request.get_data()
    image_array = bytes_to_array(image_bytes)

    apply_mask = DEFAULT_APPLY_MASK
    smooth = True
    palette = DEFAULT_CMAP

    inference_class = inference_classes_dict[VALUE_STABLE]
    output = inference_class.prepare_output(
        image_array, apply_mask=apply_mask, smooth=smooth, cmap_name=palette, output_json=True
    )

    total_time = time.time() - start_time
    print(f'Image processed in {total_time:.3f} seconds')

    return jsonify(output), 200


if __name__ == '__main__':
    port = 8500
    print(f'The app is running on port {port}')
    app.run(port=port)
