import argparse
import time

from beauty_model import BeautyModel
from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Train face beauty model')
    parser.add_argument('-c', '--config', type=str, default='default_config.json', help='Path to a train config')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = load_json(args.config)
    print('\nconfig:')
    from pprint import pprint
    pprint(config)

    start_time = time.time()

    beauty_model = BeautyModel(config)
    beauty_model.create_models()
    beauty_model.train_top_model_from_config()
    beauty_model.train_full_model_from_config()

    total_time = time.time() - start_time
    print(f'\n\n\nModels trained in {total_time:.3f} seconds')
