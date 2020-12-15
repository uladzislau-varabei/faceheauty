import os
from copy import deepcopy
import json
import time

import cv2
import dlib
import numpy as np
import tensorflow as tf

from dataset_utils import preprocess_image
from model_v1_utils import load_model
from viz_utils import create_gradcam_image, create_gradcam_models, make_gradcam_heatmap
from utils import load_config, tf_resize_image, DEFAULT_CMAP


def dlib_ffd_extract_faces(image, detector=None, expand_face_area=True):
    # Note: faces detected with dlib frontal detector are badly aligned

    if detector is None:
        detector = dlib.get_frontal_face_detector()

    # dlib frontal face detector example: http://dlib.net/face_detector.py.html
    # Note: scores can be greater than 1.0 (yet preds seem to be sorted by scores),
    # so they are not compared with threshold
    preds, scores, _ = detector.run(image, 1)

    if len(preds) == 0:
        return [(np.zeros([100, 100, 3], dtype=np.uint8), False)]

    h, w, c = image.shape
    faces_results = []

    for pred in preds:
        h_max = pred.bottom()
        h_min = pred.top()
        w_min = pred.left()
        w_max = pred.right()

        if expand_face_area:
            h_range = h_max - h_min
            w_range = w_max - w_min

            # Top part of image
            new_h_min = max(0, h_min - int(0.35 * h_range))
            # Bottom part of image
            new_h_max = min(h, h_max + int(0.15 * h_range))
            new_w_min = max(0, w_min - int(0.2 * w_range))
            new_w_max = min(w, w_max + int(0.2 * w_range))

            new_h_range = new_h_max - new_h_min
            new_w_range = new_w_max - new_w_min

            face_h_min = h_min - new_h_min
            face_h_max = new_h_range - (new_h_max - h_max)
            face_w_min = w_min - new_w_min
            face_w_max = new_w_range - (new_w_max - w_max)

            face_coords = (face_h_min, face_h_max, face_w_min, face_w_max)
        else:
            new_h_max, new_h_min, new_w_min, new_w_max = h_max, h_min, w_min, w_max
            face_coords = (h_max, h_min, w_min, w_max)

        faces_results.append(
            (image[new_h_min : new_h_max, new_w_min : new_w_max, :], face_coords)
        )

    return faces_results


def dlib_cnn_extract_faces(image, detector=None):
    # Works very slow on cpu (without resizing)
    # Quality is better than that of dlib frontal detector
    # Foreheads are usually cut

    if detector is None:
        fd_path = os.path.join('dlib_models', 'mmod_human_face_detector.dat')
        dnn_face_detector = dlib.cnn_face_detection_model_v1(fd_path)
        detector = dnn_face_detector

    faces_results = []
    preds = detector(image)
    for pred in preds:
        w_min = pred.rect.left()
        w_max = pred.rect.right()
        h_max = pred.rect.bottom()
        h_min = pred.rect.top()
        faces_results.append(image[h_min:h_max, w_min:w_max, :])

    if len(faces_results) == 0:
        print('No faces detected')

    return faces_results


def cv2_detect_faces(input_image, cv2_face_model=None, source_rgb=True):
    # Note: there is one additional check in cv2_cnn_extract_faces
    if cv2_face_model is None:
        prototxt_path = os.path.join('opencv_models', 'deploy.prototxt.txt')
        model_path = os.path.join('opencv_models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        cv2_face_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    image = deepcopy(input_image)
    if source_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    cv2_face_model.setInput(blob)
    detections = cv2_face_model.forward()

    faces_coords = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        font_scale = 1.0
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # draw the bounding box of the face along with the associated probability
            text = '{:.2f}%'.format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

            face_coords = (startY, endY, startX, endX)
            faces_coords.append(face_coords)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, faces_coords


def cv2_cnn_extract_faces(input_image, cv2_face_model=None, source_rgb=True, expand_face_area=True):
    if cv2_face_model is None:
        prototxt_path = os.path.join('opencv_models', 'deploy.prototxt.txt')
        model_path = os.path.join('opencv_models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        cv2_face_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = deepcopy(input_image)
    if source_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    cv2_face_model.setInput(blob)
    detections = cv2_face_model.forward()

    faces_results = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.5:
            print('cv2 face confidence:', confidence)

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            face_coords = (startY, endY, startX, endX)
            h_min, h_max, w_min, w_max = face_coords

            # Sometimes ranges are strange, so this step is required
            # (increased confidence threshold also doesn't always help)
            x_cond = (0 <= startX <= w) and (0 <= endX <= w) and (startX < endX)
            y_cond = (0 <= startY <= h) and (0 <= endY <= h) and (startY < endY)
            # Also skip very small faces when at least one has been extracted (remember images size in training)
            if len(faces_results) == 0:
                size_cond = True
            else:
                size_cond = (endX - startX) > 100 and (endY - startY) > 100
            range_cond = x_cond and y_cond and size_cond

            if range_cond:
                if expand_face_area:
                    h_range = h_max - h_min
                    w_range = w_max - w_min

                    # Top part of image
                    new_h_min = max(0, h_min - int(0.2 * h_range))
                    # Bottom part of image
                    new_h_max = min(h, h_max + int(0.1 * h_range))
                    # Not sure which will be better for heatmaps: 0.15 or 0.2
                    new_w_min = max(0, w_min - int(0.2 * w_range))
                    new_w_max = min(w, w_max + int(0.2 * w_range))

                    new_h_range = new_h_max - new_h_min
                    new_w_range = new_w_max - new_w_min

                    face_h_min = h_min - new_h_min
                    face_h_max = new_h_range - (new_h_max - h_max)
                    face_w_min = w_min - new_w_min
                    face_w_max = new_w_range - (new_w_max - w_max)

                    face_coords = (face_h_min, face_h_max, face_w_min, face_w_max)
                else:
                    new_h_max, new_h_min, new_w_min, new_w_max = h_max, h_min, w_min, w_max
                    face_coords = (h_min, h_max, w_min, w_max)

                faces_results.append(
                    (to_rgb(image[new_h_min : new_h_max, new_w_min : new_w_max, :]), face_coords)
                )

    return faces_results


def json_to_array(data):
    return np.array(json.loads(data), dtype=np.uint8)


def to_json_response(x):
    return {'heatmap_image': json.dumps(x[0].tolist()), 'message': x[1]}


def convert_from_json_response(r):
    r_json = json.loads(r.content)
    outputs = []
    for out in r_json['output']:
        outputs.append([
            json_to_array(out['heatmap_image']),
            out['message']
        ])
    return outputs


class InferenceBeautyApp:

    def __init__(self, model_path, model=None):
        self.model_path = model_path
        self.config = load_config(self.model_path)
        self.image_target_size = self.config['image_target_size']
        self.backbone = self.config['backbone']
        self.normalize_embeddings = self.config['normalize_embeddings']
        self.pooling = self.config['pooling']
        self.n_outputs = self.config['n_outputs']

        if model is None:
            model = load_model(model_path)
        self.model = model
        self.create_gradcam_models()

        self.load_cv2_face_model()

    def load_cv2_face_model(self):
        prototxt_path = os.path.join('opencv_models', 'deploy.prototxt.txt')
        model_path = os.path.join('opencv_models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        self.cv2_face_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def create_gradcam_models(self):
        base_model = self.model.layers[0]
        top_model = self.model.layers[1]
        self.last_conv_layer_model, self.classifier_model = create_gradcam_models(
            base_model, top_model,
            backbone=self.backbone,
            pooling=self.pooling,
            normalize_embeddings=self.normalize_embeddings
        )

    def make_gradcam_heatmap(self, image):
        class_index = self.n_outputs - 1
        heatmap = make_gradcam_heatmap(
            image,
            last_conv_layer_model=self.last_conv_layer_model,
            classifier_model=self.classifier_model,
            class_index=class_index
        ).astype(np.float32)
        return heatmap

    def prepare_input_image(self, image):
        return tf.expand_dims(
            preprocess_image(
                tf_resize_image(image, self.image_target_size), self.backbone,
                training=False, input_preprocessing=True
            ),
            axis=[0]
        )

    def predict_score(self, image):
        score = self.model(self.prepare_input_image(image), training=False).numpy()[0]

        if score.shape[0] == 1:
            score = score[0]
        elif score.shape[0] == 2:
            score = score[1]
        else:
            score = list(score)

        return score

    def adjust_score(self, score):
        print(f'\nReal score = {score}')
        if isinstance(score, list):
            adj_score = score
        else:
            mult1 = 1.2
            mult2 = 1.15
            mult3 = 1.1
            if score <= 0.8 / mult1:
                adj_score = score * mult1
            elif score <= 0.9 / mult2:
                adj_score = score * mult2
            elif score <= 0.99 / mult3:
                adj_score = score * mult3
            else:
                adj_score = score
        print(f'Adjusted score = {adj_score}')
        return adj_score

    def prepare_message_based_on_score(self, score):
        message = ''
        if score < 0.5:
            message = 'The face is normal'
        elif score < 0.8:
            message = 'The face is beautiful'
        elif score <= 1.0:
            message = 'The face is outstanding'
        message = f'Beauty score: {score:.2f}' \
                  f'\n{message}'
        return message

    def prepare_output_for_single_face_image(self, face_result, apply_mask=True, smooth=True, cmap_name=DEFAULT_CMAP):
        # Note: face coords are used to hide heatmap outside of detected face
        # as face image is larger then result of detector
        face_image, face_coords = face_result
        score = self.predict_score(face_image)
        score = self.adjust_score(score)
        message = self.prepare_message_based_on_score(score)
        input_face_image = self.prepare_input_image(face_image)
        heatmap = self.make_gradcam_heatmap(input_face_image)
        mask_coords = face_coords if apply_mask else None
        heatmap_image = np.array(
            create_gradcam_image(face_image / 255., heatmap, mask_coords, smooth, cmap_name=cmap_name)
        )
        return heatmap_image, message

    def convert_output_to_json(self, output):
        return {'output': [to_json_response(out) for out in output]}

    def prepare_output(self, image, apply_mask=True, smooth=True, cmap_name=DEFAULT_CMAP, face_detect_fun=None,
                       output_json=False):
        """
        image - np.array of shape (h, w, c) with dtype=np.uint8
        """
        if face_detect_fun is None:
            face_detect_fun = lambda image: cv2_cnn_extract_faces(image, self.cv2_face_model)

        faces_results = face_detect_fun(image)

        output = [
            self.prepare_output_for_single_face_image(face_result, apply_mask, smooth, cmap_name)
            for face_result in faces_results
        ]
        if len(output) == 0:
            output = [
                (
                    np.zeros((100,100,3), dtype=np.uint8),
                    'No face detected on the image. Please, upload another image'
                )
            ]

        if output_json:
            start_time = time.time()
            output = self.convert_output_to_json(output)
            total_time = time.time() - start_time
            print(f'Output converted to json in {total_time:.3f} seconds')

        return output
