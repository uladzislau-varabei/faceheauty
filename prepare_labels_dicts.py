import os
import glob
import json


def get_real_image_path(p, images_paths):
    real_path = None
    for img_p in images_paths:
        if p in img_p:
            real_path = img_p
            break
    return real_path


def load_labels_data():
    labels_path = os.path.join('SCUT-FBP5500_v2', 'train_test_files', 'All_labels.txt')
    with open(labels_path, 'r') as f:
        labels_data = f.readlines()
    return labels_data


def create_labels_dict(labels_data):
    labels_dict = {}
    for l in labels_data:
        fname, score = l.split()
        labels_dict[get_real_image_path(fname, images_paths)] = float(score)
    return labels_dict


def score_mapping(s):
    if 0.0 <= s < 2.0:
        return 1
    if 2.0 <= s < 2.7:
        return 2
    if 2.7 <= s < 3.25:
        return 3
    if 3.25 <= s < 4.0:
        return 4
    if 4.0 <= s < 5.0:
        return 5


def create_int_labels_dict(labels_dict):
    int_labels_dict = {k: score_mapping(v) for k, v in labels_dict.items()}
    return int_labels_dict


if __name__ == '__main__':
    images_paths = glob.glob(os.path.join('SCUT-FBP5500_v2', 'Images', '*'))

    labels_data = load_labels_data()
    labels_dict = create_labels_dict(labels_data)
    int_labels_dict = create_int_labels_dict(labels_dict)

    with open('labels.json', 'w') as fp:
        json.dump(labels_dict, fp)

    with open('int_labels.json', 'w') as fp:
        json.dump(int_labels_dict, fp)
