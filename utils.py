import os
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt


# ---------- Tf utils ----------

def initialize_gpu(mode='growth', memory_limit=1024):
    assert mode in ['growth', 'limit']
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if mode == 'growth':
                print('Memory growth mode')
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            elif mode == 'limit':
                print(f'Memory limit mode. Limit = {memory_limit}')
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)
                        ]
                    )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print('GPU is not available')


def tf_resize_image(image, size):
    return tf.image.resize(image, size, method=tf.image.ResizeMethod.LANCZOS3)


def l2_normalizer():
    return tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='Embeddings')


def get_loss_fn(n_outputs, loss_name):
    assert loss_name in ['CE', 'MSE']
    if loss_name == 'MSE':
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        if n_outputs == 1:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return loss_fn


def get_optimizer(optimizer_name, optimizer_lr):
    return {
        'adam': tf.keras.optimizers.Adam(optimizer_lr),
        'sgd': tf.keras.optimizers.SGD(optimizer_lr, momentum=0.9)
    }[optimizer_name.lower()]


def add_l2_weights_decay(model, weight_decay):
    # Thanks to https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras (mathmanu)
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    # Recursive call for nested models
    def add_decay_loss(m, factor):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with tf.keras.backend.name_scope('weight_regularizer'):
                    regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
                    m.add_loss(regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay / 2.0)
    return


class DelayedEarlyStoppingCallback(tf.keras.callbacks.Callback):
    # In addition to EarlyStoppingCallback this class allows one to monitor stop condition only after some
    # number of steps (delay arg)
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 delay=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(DelayedEarlyStoppingCallback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.delay = delay
        self.delayed_phase = True
        self.n_empty_calls = 0
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
          self.monitor_op = np.less
        elif mode == 'max':
          self.monitor_op = np.greater
        else:
          if 'acc' in self.monitor:
            self.monitor_op = np.greater
          else:
            self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.delayed_phase = True
        self.n_empty_calls = 0

    def on_epoch_end(self, epoch, logs=None):
        # Additional code for delay
        if self.delayed_phase:
            self.n_empty_calls += 1
            if self.n_empty_calls == self.delay:
                self.delayed_phase = False
        # Original callback code
        else:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print('Restoring model weights from the end of the best epoch.')
                        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('Early stopping conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        return monitor_value


class DelayedReduceLROnPlateau(tf.keras.callbacks.Callback):
    # In addition to ReduceLROnPlateauCallback this class allows one to monitor reduce LR condition only after some
    # number of steps (delay arg)
    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 delay=0,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(DelayedReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            print('`epsilon` argument is deprecated and will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.delay = delay
        self.delayed_phase = True
        self.n_empty_calls = 0
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Learning Rate Plateau Reducing mode %s is unknown, '
                  'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        # Additional code for delay
        if self.delayed_phase:
            self.n_empty_calls += 1
            if self.n_empty_calls == self.delay:
                self.delayed_phase = False
        # Original callback code
        else:
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            current = logs.get(self.monitor)
            if current is None:
                print('Reduce LR on plateau conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                      'rate to %s.' % (epoch + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


# ---------- Visualizing utils ----------

def visualize(img, hide_axis=False):
    plt.figure(1)
    if hide_axis:
        plt.axis('off')
    plt.imshow(img)
    plt.show()


def plot_training_history(history, start_val=1):
    hist_train_loss = history['loss']
    hist_valid_loss = history['val_loss']

    hist_train_mae = history['mean_absolute_error']
    hist_valid_mae = history['val_mean_absolute_error']

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 8), nrows=1, ncols=2, dpi=150)

    ax1.plot(hist_train_loss[start_val:], label='Train')
    ax1.plot(hist_valid_loss[start_val:], label='Valid')
    ax1.set_title('Loss', fontsize=20)
    ax1.grid('--', linewidth=0.5, alpha=0.5)
    ax1.legend(fontsize=18)

    ax2.plot(hist_train_mae[start_val:], label='Train')
    ax2.plot(hist_valid_mae[start_val:], label='Valid')
    ax2.set_title('MAE', fontsize=20)
    ax2.grid('--', linewidth=0.5, alpha=0.5)
    ax2.legend(fontsize=18)

    plt.show()


def plot_training_history_from_file(fpath, start_val=1):
    with open(fpath, 'r') as fp:
        history = json.load(fp)

    hist_train_loss = history['history']['loss']
    hist_valid_loss = history['history']['val_loss']

    hist_train_mae = history['history']['mean_absolute_error']
    hist_valid_mae = history['history']['val_mean_absolute_error']

    test_cond = 'test_scores' in history.keys()
    if test_cond:
        test_loss = history['test_scores']['loss']
        test_mae = history['test_scores']['mean_absolute_error']

    plot_label = os.path.split(history['model_name'])[1]

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 8), nrows=1, ncols=2, dpi=200)
    fig.suptitle(plot_label, fontsize=20)

    ax1.plot(hist_train_loss[start_val:], label='Train')
    ax1.plot(hist_valid_loss[start_val:], label='Valid')
    if test_cond:
        ax1.scatter(len(hist_train_loss) - start_val - 1, test_loss, label='Test', c='red', s=50)
    ax1.set_title('Loss', fontsize=20)
    ax1.grid('--', linewidth=0.5, alpha=0.5)
    ax1.legend(fontsize=18)

    ax2.plot(hist_train_mae[start_val:], label='Train')
    ax2.plot(hist_valid_mae[start_val:], label='Valid')
    if test_cond:
        ax2.scatter(len(hist_train_mae) - start_val - 1, test_mae, label='Test', c='red', s=50)
    ax2.set_title('MAE', fontsize=20)
    ax2.grid('--', linewidth=0.5, alpha=0.5)
    ax2.legend(fontsize=18)

    plt.show()


def add_title_background(img_array, title_height=None):
    h, w, _ = img_array.shape
    if title_height is None:
        title_height = int(0.1 * h)
    background = np.zeros([title_height, w, 3], dtype=img_array.dtype)
    return np.vstack([background, img_array])


def convert_to_pil_image(images):
    """
    :param images: numpy array of dtype=uint8 in range [0, 255]
    :return: PIL image
    """
    return Image.fromarray(images)


def convert_to_pil_image_with_title(img_array, title, title_height=None, title_font_size=None):
    h, w, _ = img_array.shape
    #img_array = (255 * add_title_background(img_array)).astype(np.uint8)
    img_array = add_title_background(img_array, title_height)
    img = convert_to_pil_image(img_array)

    # See function add_title_background
    if title_font_size is None:
        title_font_size = int(2 * 0.04 * h)
    # Font can be stored in a folder with script
    font = ImageFont.truetype('arial.ttf', title_font_size)

    d = ImageDraw.Draw(img)
    # text_w, text_h = d.textsize(title)
    text_w_start_pos = (w - font.getsize(title)[0]) / 2
    d.text((text_w_start_pos, 0.01 * h), title, fill='white', font=font)
    return img


def fast_make_grid(images, nrows=None, ncols=None, padding=None, all_images=True):
    """
    all_images - plot all_images? Only matters for cases when last row of grid is not complete,
    i.e. there will be blank areas
    """
    if padding is None:
        padding = 5

    if (nrows is not None) and (ncols is None):
        ncols = len(images) // nrows
        if all_images:
            if nrows * ncols < len(images):
                ncols += 1
    elif (nrows is None) and (ncols is not None):
        nrows = len(images) // ncols
        if all_images:
            if nrows * ncols < len(images):
                nrows += 1
    elif (nrows is None) and (ncols is None):
        assert False

    _, h, w, _ = images.shape
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncols * w + (ncols - 1) * padding

    image_grid = np.zeros((grid_h, grid_w, 3), dtype=images.dtype)
    hp = h + padding
    wp = w + padding

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            image_grid[hp * r: hp * (r + 1) - padding, wp * c: wp * (c + 1) - padding, :] = images[i]
            i += 1
            if i == len(images):
                break
        if i == len(images):
            break

    return image_grid


def fast_save_grid(out_dir, fname, images, nrows, ncols, padding=None,
                   title=None, title_height=None, title_font_size=None, save_in_jpg=False):
    img_grid = fast_make_grid(images, nrows=nrows, ncols=ncols, padding=padding)
    if title is not None:
        img = convert_to_pil_image_with_title(img_grid, title, title_height=title_height, title_font_size=title_font_size)
    else:
        img = convert_to_pil_image(img_grid)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if save_in_jpg:
        img.save(os.path.join(out_dir, fname + '.jpg'), 'JPEG', quality=95, optimize=True)
    else:
        img.save(os.path.join(out_dir, fname + '.png'), 'PNG')


def plot_color_gradients(cmap_list, fname=None):
    # Thanks to https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    nrows = len(cmap_list)
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title('Available colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()

    if fname is not None:
        if fname.split('.')[-1].lower() not in ['jpeg', 'jpg']:
            fname = fname + '.jpeg'
        plt.savefig(fname, bbox_inches='tight', dpi=200, pil_kwargs={'quality': 95})
    else:
        plt.show()


# ---------- Other utils ----------

def load_config(model_path):
    p_head, _ = model_path.rsplit('_', 1)
    config_path = 'configs' + p_head[6:] + '.json'
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def numpy_dict_to_json_format(d):
    def convert_value(v):
        if isinstance(v, list):
            return [float(x) for x in v]
        else:
            return v
    return {k: convert_value(v) for k, v in d.items()}


def load_json(p):
    with open(p, 'r') as fp:
        data = json.load(fp)
    return data


def dump_json(data, fpath, **kwargs):
    p_head, _ = os.path.split(fpath)
    if len(p_head) > 0:
        if not os.path.exists(p_head):
            os.makedirs(p_head, exist_ok=True)
    with open(fpath + '.json', 'w') as fp:
        json.dump(data, fp, **kwargs)
    print(f'Saved to {fpath}')


def select_images_group(images_paths, groups=None):
    if groups is None:
        # AF - Asian female
        # AM - Asian male
        # CF - Caucasian female
        # CM - Caucasian male
        groups = ['AF', 'AM', 'CM', 'CF']
    return [p for p in images_paths if os.path.split(p)[1][:2] in groups]


def prepare_target_variable_dict(n_outputs, max_value, labels_dict, int_labels_dict):
    def create_one_hot(x):
        zero_arr = np.zeros(n_outputs, dtype=np.int64)
        zero_arr[x - 1] = 1
        # Array are meant to be saved in jsons, so convert to list
        return list(zero_arr)

    def to_proba_vector(v):
        p = v / max_value
        return np.array([1. - p, p], dtype=np.float32)

    # labels_dict - floats from 0. to 5.
    # int_labels_dict - ints from 1 to 5
    if n_outputs == 1:
        return {k: v / max_value for k, v in labels_dict.items()}
    elif n_outputs == 2:
        return {k: to_proba_vector(v) for k, v in labels_dict.items()}
    else:
        return {k: create_one_hot(v) for k, v in int_labels_dict.items()}


def create_intervals(points):
    intervals = []
    for idx in range(len(points) - 1):
        intervals.append((points[idx], points[idx + 1]))
    return intervals


def adjusted_train_test_split(paths, labels_dict, test_size):
    max_score = np.ceil(max(list(labels_dict.values())))
    interval_points = np.arange(0., max_score + 0.1, 0.5)
    groups_dict = {k: [] for k in create_intervals(interval_points)}

    for p in paths:
        score = labels_dict[p]
        for gr in groups_dict.keys():
            if gr[0] <= score <= gr[1]:
                groups_dict[gr].append(p)
                break

    train_split = []
    test_split = []
    for gr, gr_paths in groups_dict.items():
        if len(gr_paths) == 1:
            train_split += gr_paths
        elif len(gr_paths) > 1:
            gr_train_split, gr_test_split = train_test_split(gr_paths, test_size=test_size, random_state=42)
            train_split += gr_train_split
            test_split += gr_test_split

    return train_split, test_split


# ---------- Web app constants ----------

KEY_MODEL = 'model'
KEY_APPLY_MASK = 'apply_mask'
KEY_PALETTE = 'palette'

VALUE_STABLE = 'Stable'
VALUE_EXPERIMENTAL = 'Experimental'
VALUE_YES = 'Yes'
VALUE_NO = 'No'

CMAP_CIVIDIS = 'cividis'
CMAP_VIRIDIS = 'viridis'
CMAP_INFERNO = 'inferno'
CMAP_MAGMA = 'magma'
CMAP_PURD = 'PuRd'
CMAP_HOT = 'hot'
CMAP_AFMHOT = 'afmhot'
CMAP_GISTHEAT = 'gist_heat'
CMAP_COOLWARM = 'coolwarm'
CMAP_BWR = 'bwr'
CMAP_JET = 'jet'

ALL_CMAPS = [
    CMAP_CIVIDIS, CMAP_VIRIDIS, CMAP_INFERNO, CMAP_MAGMA, CMAP_PURD,
    CMAP_HOT, CMAP_AFMHOT, CMAP_GISTHEAT, CMAP_COOLWARM, CMAP_BWR, CMAP_JET
]

DEFAULT_APPLY_MASK = VALUE_YES
DEFAULT_MODEL = VALUE_STABLE
DEFAULT_CMAP = CMAP_JET
