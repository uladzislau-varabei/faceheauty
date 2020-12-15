import os
import glob
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa

from dataset_utils import create_dataset
from model_v1_utils import get_base_model, create_top_model, create_model, create_bottleneck_features
from utils import initialize_gpu, add_l2_weights_decay, get_loss_fn, get_optimizer, \
    DelayedEarlyStoppingCallback, DelayedReduceLROnPlateau, \
    select_images_group, prepare_target_variable_dict, adjusted_train_test_split, \
    dump_json, numpy_dict_to_json_format


def create_model_name(model_type, model_version, backbone, normalize_embeddings, image_size,
                      tl_workflow, images_groups, n_outputs, loss_name,
                      top_model_opt, top_model_lr,
                      full_model_opt=None, full_model_lr=None,
                      full_model_decay=None,
                      full_model_train_batchnorm=None, ftune_model_postfix=None):
    def join_into_postfix(groups):
        if groups is not None:
            return ''.join(sorted(groups))
        else:
            return 'ALL'

    def format_float(x, to_decimal=False):
        # By default keep standard preprocessing
        if to_decimal:
            str_x = ('%.17f' % x)
            str_idx = 0
            for idx in range(1, len(str_x) + 1):
                if str_x[-idx] != '0':
                    str_idx = idx
                    if str_x[-idx] == '.':
                        str_idx -= 1
                    break
            return str_x[:-str_idx + 1]
        else:
            return x

    assert model_type in ['top', 'full']

    if backbone.lower() in ['facenet', 'face_evoLVe_ir50'.lower(), 'insightface']:
        norm_embeds = '_normembeds' if normalize_embeddings else '_nonormembeds'
    else:
        norm_embeds = ''

    model_name = f'v{model_version}_{model_type}'\
                 f'_{backbone}{norm_embeds}_size{image_size}_{tl_workflow}'\
                 f'_data{join_into_postfix(images_groups)}_out{n_outputs}_{loss_name}'\
                 f'_{top_model_opt}{format_float(top_model_lr)}'
    if model_type == 'full':
        full_model_postfix = f'_{full_model_opt}{format_float(full_model_lr)}'
        decay_postfix = f'_decay{format_float(full_model_decay)}' if full_model_decay is not None else '_nodecay'
        full_model_postfix += decay_postfix
        model_name += full_model_postfix
        if full_model_train_batchnorm is not None:
            bn_postfix = '_trBN' if full_model_train_batchnorm else '_fzBN'
            model_name += bn_postfix
        if ftune_model_postfix is not None:
            model_name += ftune_model_postfix

    dir_prefix = os.path.join(f'v{model_version}', f'{model_type}')

    return os.path.join(dir_prefix, model_name)


class BeautyModel:

    def __init__(self, config):
        self.config = config
        # Dataset related part
        self.images_paths = sorted(glob.glob(os.path.join('SCUT-FBP5500_v2', 'Images', '*')))
        self.images_groups = config['images_groups']
        self.images_paths = select_images_group(self.images_paths, self.images_groups)
        self.image_target_size = config['image_target_size']
        self.max_score = config['max_score']

        # Model related part
        self.use_mixed_precision = config['use_mixed_precision']
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.compute_dtype = 'float16' if self.use_mixed_precision else 'float32'
        self.input_shape = tuple(self.image_target_size) + (3,)
        self.backbone = config['backbone']
        self.pooling = config['pooling']
        self.n_outputs = config['n_outputs']
        self.n_layers = config['layers']
        self.units = config['units']
        self.act = config['act']
        self.use_batchnorm = config['use_batchnorm']
        self.use_dropout = config['use_dropout']
        self.dropout_rate = config['dropout_rate']
        self.loss_name = config['loss_name']
        self.normalize_embeddings = config['normalize_embeddings']
        self.model_version = config['model_version']
        self.use_test_split = config['use_test_split']

        # Stable: freeze base model part and pass images through the whole network
        # Fast: compute bottleneck features (no augmentations) and only pass features through the top model
        self.tl_workflow = config['tl_workflow']
        assert self.tl_workflow in ['stable', 'fast']

        self.ftune_batchnorm = config['ftune_batchnorm']
        self.ftune_train_patterns = config['ftune_train_patterns']
        self.ftune_model_postfix = config['ftune_model_postfix']

        self.tl_train_params = config.get('tl_train_params', dict())
        if len(self.tl_train_params.keys()) > 0:
            self.tl_epochs = self.tl_train_params['epochs']
            self.tl_optimizer_name = self.tl_train_params['optimizer_name']
            self.tl_optimizer_lr = self.tl_train_params['optimizer_lr']
            self.tl_early_stop_cb = self.tl_train_params['early_stop_cb']
            self.tl_reduce_lr_cb = self.tl_train_params['reduce_lr_cb']
            self.tl_tboard_cb = self.tl_train_params['tboard_cb']
            self.tl_save_last_model = self.tl_train_params['save_last_model']
            self.tl_jupyter_mode = self.tl_train_params['jupyter_mode']

        self.ftune_train_params = config.get('ftune_train_params', dict())
        if len(self.ftune_train_params.keys()) > 0:
            self.ftune_epochs = self.ftune_train_params['epochs']
            self.ftune_optimizer_name = self.ftune_train_params['optimizer_name']
            self.ftune_optimizer_lr = self.ftune_train_params['optimizer_lr']
            self.ftune_weights_decay = self.ftune_train_params['weights_decay']
            self.ftune_early_stop_cb = self.ftune_train_params['early_stop_cb']
            self.ftune_reduce_lr_cb = self.ftune_train_params['reduce_lr_cb']
            self.ftune_tboard_cb = self.ftune_train_params['tboard_cb']
            self.ftune_save_intermediate_models = self.ftune_train_params['save_intermediate_models']
            self.ftune_save_last_model = self.ftune_train_params['save_last_model']
            self.ftune_jupyter_mode = self.ftune_train_params['jupyter_mode']

        self.prepare_target_variable_dict()
        self.split_paths()
        self.create_datasets()


    def create_model_name(self, model_type, top_model_opt, top_model_lr,
                          full_model_opt=None, full_model_lr=None, full_model_decay=None):
        return create_model_name(
            model_type,
            model_version=self.model_version,
            backbone=self.backbone,
            normalize_embeddings=self.normalize_embeddings,
            image_size=self.image_target_size[0],
            tl_workflow=self.tl_workflow,
            images_groups=self.images_groups,
            n_outputs=self.n_outputs,
            loss_name=self.loss_name,
            top_model_opt=top_model_opt,
            top_model_lr=top_model_lr,
            full_model_opt=full_model_opt,
            full_model_lr=full_model_lr,
            full_model_decay=full_model_decay,
            full_model_train_batchnorm=self.ftune_batchnorm,
            ftune_model_postfix=self.ftune_model_postfix
        )

    def prepare_target_variable_dict(self):
        with open('labels.json', 'r') as fp:
            self.labels_dict = json.load(fp)

        with open('int_labels.json', 'r') as fp:
            self.int_labels_dict = json.load(fp)

        self.target_variable_dict = prepare_target_variable_dict(
            self.n_outputs, self.max_score, labels_dict=self.labels_dict, int_labels_dict=self.int_labels_dict
        )

    def split_paths(self):
        if self.use_test_split:
            self.train_paths, self.test_valid_paths = adjusted_train_test_split(
                self.images_paths, self.labels_dict, test_size=0.3
            )
            self.valid_paths, self.test_paths = adjusted_train_test_split(
                self.test_valid_paths, self.labels_dict, test_size=0.5
            )
        else:
            self.train_paths, self.valid_paths = adjusted_train_test_split(
                self.images_paths, self.labels_dict, test_size=0.2
            )

    def create_models(self):
        initialize_gpu()
        tf.keras.backend.clear_session()
        self.base_model = get_base_model(
            self.backbone, self.pooling, self.input_shape, normalize_embeddings=self.normalize_embeddings
        )
        self.base_model.trainable = False
        self.top_model = create_top_model(
            n_outputs=self.n_outputs,
            n_layers=self.n_layers,
            units=self.units,
            act=self.act,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout,
            dropout_rate=self.dropout_rate,
            input_shape=(self.base_model.output_shape[1],)
        )
        self.full_model = tf.keras.Sequential([self.base_model, self.top_model])

    def create_datasets(self):
        # Needed for model saving callback
        self.train_batch_size = 32
        batch_size = 32

        self.train_ds = create_dataset(
            self.train_paths, target_size=self.image_target_size, backbone=self.backbone,
            training=True, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
            batch_size=self.train_batch_size
        )
        self.valid_ds = create_dataset(
            self.valid_paths, target_size=self.image_target_size, backbone=self.backbone,
            training=False, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
            batch_size=batch_size
        )
        if self.use_test_split:
            self.test_ds = create_dataset(
                self.test_paths, target_size=self.image_target_size, backbone=self.backbone,
                training=False, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
                batch_size=batch_size
            )

        if self.tl_workflow == 'fast':
            self.train_bottleneck_ds = create_dataset(
                self.train_paths, target_size=self.image_target_size, backbone=self.backbone,
                training=False, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
                batch_size=batch_size
            )
            self.valid_bottleneck_ds = create_dataset(
                self.valid_paths, target_size=self.image_target_size, backbone=self.backbone,
                training=False, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
                batch_size=batch_size
            )
            if self.use_test_split:
                self.test_bottleneck_ds = create_dataset(
                    self.test_paths, target_size=self.image_target_size, backbone=self.backbone,
                    training=False, input_preprocessing=True, target_variable_dict=self.target_variable_dict,
                    batch_size=batch_size
                )

    def create_bottleneck_data(self, jupyter_mode=False, force=False):
        self.bottleneck_data_ready = False

        if (not self.bottleneck_data_ready) or force:
            self.create_datasets()

            print('Bottleneck train dataset:')
            self.train_bottleneck_preds = create_bottleneck_features(
                self.base_model, self.train_bottleneck_ds, len(self.train_paths),
                dtype=self.compute_dtype, jupyter_mode=jupyter_mode
            )
            print('Bottleneck valid dataset:')
            self.valid_bottleneck_preds = create_bottleneck_features(
                self.base_model, self.valid_bottleneck_ds, len(self.valid_paths),
                dtype=self.compute_dtype, jupyter_mode=jupyter_mode
            )
            if self.use_test_split:
                print('Bottleneck test dataset:')
                self.test_bottleneck_preds = create_bottleneck_features(
                    self.base_model, self.test_bottleneck_ds, len(self.test_paths),
                    dtype=self.compute_dtype, jupyter_mode=jupyter_mode
                )

            self.train_bottleneck_target = np.array([self.target_variable_dict[p] for p in self.train_paths])
            self.valid_bottleneck_target = np.array([self.target_variable_dict[p] for p in self.valid_paths])
            if self.use_test_split:
                self.test_bottleneck_target = np.array([self.target_variable_dict[p] for p in self.test_paths])

            self.bottleneck_data_ready = True
            print('Bottleneck data ready!')

    def prepare_top_model_callbacks(self, model_name, early_stop_cb=True, reduce_lr_cb=True, tboard_cb=None,
                                    jupyter_mode=False):
        # Default callbacks
        fast_tl_early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=(0.001 if self.loss_name == 'CE' else 0.), patience=40
        )
        stable_tl_early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=(0.00025 if self.loss_name == 'CE' else 0.), patience=50
        )
        fast_tl_reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.33, patience=10, min_lr=0.0001,
        )
        stable_tl_reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.33, patience=20, min_lr=0.00001,
        )
        tqdm_callback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False)
        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', model_name), histogram_freq=0, write_graph=True,
            update_freq='epoch', profile_batch=2
        )

        # Callbacks setting
        callbacks = []
        if jupyter_mode:
            callbacks.append(tqdm_callback)
        if (tboard_cb is not None) and (tboard_cb is not False):
            tboard_callback = tboard_callback if tboard_cb is True else tboard_cb
            callbacks.append(tboard_callback)
        if early_stop_cb is not None:
            if early_stop_cb is True:
                if self.tl_workflow == 'fast':
                    early_stop_callback = fast_tl_early_stop_callback
                else:
                    early_stop_callback = stable_tl_early_stop_callback
            else:
                early_stop_callback = early_stop_cb
            callbacks.append(early_stop_callback)
        if reduce_lr_cb is not None:
            if reduce_lr_cb is True:
                if self.tl_workflow == 'fast':
                    reduce_lr_callback = fast_tl_reduce_lr_callback
                else:
                    reduce_lr_callback = stable_tl_reduce_lr_callback
            else:
                reduce_lr_callback = reduce_lr_cb
            callbacks.append(reduce_lr_callback)

        self.top_model_callbacks = callbacks

    def prepare_full_model_callbacks(self, model_name, early_stop_cb=True, reduce_lr_cb=True, tboard_cb=None,
                                     save_intermediate_models=False, save_models_path=None,
                                     jupyter_mode=False):
        callbacks = []
        if jupyter_mode:
            tqdm_callback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False)
            callbacks.append(tqdm_callback)
        if (tboard_cb is not None) and (tboard_cb is not False):
            if tboard_cb is True:
                tboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join('logs', model_name), histogram_freq=0, write_graph=True,
                    update_freq='epoch', profile_batch=2
                )
            else:
                tboard_callback = tboard_cb
            callbacks.append(tboard_callback)
        if early_stop_cb is not None:
            if early_stop_cb is True:
                # Always train for delay epochs and only then start accumulating statistics and
                # checking stop condition
                early_stop_callback = DelayedEarlyStoppingCallback(
                    monitor='val_loss', min_delta=(0.00005 if self.loss_name == 'CE' else 0.), patience=100, delay=50
                )
            else:
                early_stop_callback = early_stop_cb
            callbacks.append(early_stop_callback)
        if reduce_lr_cb is not None:
            if reduce_lr_cb is True:
                reduce_lr_callback = DelayedReduceLROnPlateau(
                    monitor='val_loss', factor=0.33, patience=25, min_lr=1e-8, delay=25
                )
            else:
                reduce_lr_callback = reduce_lr_cb
            callbacks.append(reduce_lr_callback)
        if save_intermediate_models and save_models_path is not None:
            save_every_epochs = 100
            save_freq = save_every_epochs * (len(self.train_paths) // self.train_batch_size)
            save_model_callback = tf.keras.callbacks.ModelCheckpoint(
                save_models_path + '_epoch{epoch:03d}', save_freq=save_freq
            )
            callbacks.append(save_model_callback)

        self.full_model_callbacks = callbacks

    def train_top_model(self, epochs=None, optimizer_name='adam', optimizer_lr=0.001,
                        early_stop_cb=True, reduce_lr_cb=True, tboard_cb=None, save_last_model=True,
                        jupyter_mode=False):
        if epochs is None:
            epochs = 100 if self.tl_workflow == 'fast' else 300
        batch_size = 32

        self.top_model_opt = optimizer_name
        self.top_model_lr = optimizer_lr
        model_name = self.create_model_name(
            'top', top_model_opt=self.top_model_opt, top_model_lr=self.top_model_lr
        )

        # Save model information for inference
        dump_json(self.config, os.path.join('configs', model_name), indent=4)

        # Compile options
        # Fit automatically scales loss
        top_model_optimizer = get_optimizer(optimizer_name, optimizer_lr)
        loss_fn = get_loss_fn(self.n_outputs, self.loss_name),
        metrics = tf.keras.metrics.MeanAbsoluteError()

        # Callbacks
        self.prepare_top_model_callbacks(
            model_name, early_stop_cb=early_stop_cb, reduce_lr_cb=reduce_lr_cb, tboard_cb=tboard_cb,
            jupyter_mode=jupyter_mode
        )

        if self.tl_workflow == 'fast':
            self.create_bottleneck_data(jupyter_mode)
            self.top_model.compile(optimizer=top_model_optimizer, loss=loss_fn, metrics=metrics)
            self.top_model_history = self.top_model.fit(
                x=self.train_bottleneck_preds,
                y=self.train_bottleneck_target,
                validation_data=(self.valid_bottleneck_preds, self.valid_bottleneck_target),
                epochs=epochs,
                batch_size=batch_size,
                validation_batch_size=batch_size,
                shuffle=True,
                callbacks=self.top_model_callbacks,
                verbose=0
            )
            if self.use_test_split:
                test_scores = self.top_model.evaluate(
                    self.test_bottleneck_preds, self.test_bottleneck_target, batch_size=batch_size
                )
                test_scores = dict(zip(self.top_model.metrics_names, test_scores))
        else:
            self.full_model.layers[0].trainable = False
            self.full_model.compile(optimizer=top_model_optimizer, loss=loss_fn, metrics=metrics)
            self.top_model_history = self.full_model.fit(
                x=self.train_ds,
                validation_data=self.valid_ds,
                epochs=epochs,
                # Shuffling in dataset
                shuffle=False,
                callbacks=self.top_model_callbacks,
                verbose=0
            )
            if self.use_test_split:
                test_scores = self.full_model.evaluate(self.test_ds)
                test_scores = dict(zip(self.full_model.metrics_names, test_scores))

        history_data = {
            'history': numpy_dict_to_json_format(self.top_model_history.history),
            'model_name': model_name
        }
        if self.use_test_split:
            history_data['test_scores'] = numpy_dict_to_json_format(test_scores)
        dump_json(history_data, os.path.join('history', model_name))

        if save_last_model:
            self.full_model.save(os.path.join('models', model_name), include_optimizer=False)

    def prepare_full_model_layers(self):
        def matches(name, patterns):
            return any(p in name for p in patterns)

        train_layers_patterns = self.ftune_train_patterns if self.ftune_train_patterns is not None else []

        for layer in self.base_model.layers:
            if matches(layer.name, train_layers_patterns) or len(train_layers_patterns) == 0:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    if self.ftune_batchnorm:
                        layer.trainable = True
                    else:
                        layer.trainable = False
                        print(f'Freezed batchnorm layer: {layer.name}')
                else:
                    layer.trainable = True

    def train_full_model(self, epochs=None, optimizer_name='sgd', optimizer_lr=1e-5, weights_decay=None,
                         early_stop_cb=True, reduce_lr_cb=True, tboard_cb=None,
                         save_intermediate_models=False, save_last_model=True,
                         jupyter_mode=False):
        epochs = 800 if epochs is None else epochs

        model_name = self.create_model_name(
            'full', top_model_opt=self.top_model_opt, top_model_lr=self.top_model_lr,
            full_model_opt=optimizer_name, full_model_lr=optimizer_lr, full_model_decay=weights_decay
        )
        model_path = os.path.join('models', model_name)

        # Save model information for inference
        dump_json(self.config, os.path.join('configs', model_name), indent=4)

        # Compile options
        # Fit automatically scales loss
        full_model_optimizer = get_optimizer(optimizer_name, optimizer_lr)
        loss_fn = get_loss_fn(self.n_outputs, self.loss_name),
        metrics = tf.keras.metrics.MeanAbsoluteError()

        # Callbacks
        self.prepare_full_model_callbacks(
            model_name, early_stop_cb=early_stop_cb, reduce_lr_cb=reduce_lr_cb, tboard_cb=tboard_cb,
            save_intermediate_models=save_intermediate_models, save_models_path=model_path,
            jupyter_mode=jupyter_mode
        )

        self.prepare_full_model_layers()
        add_l2_weights_decay(self.full_model, weights_decay)
        self.full_model.compile(optimizer=full_model_optimizer, loss=loss_fn, metrics=metrics)
        self.full_model_history = self.full_model.fit(
            x=self.train_ds,
            validation_data=self.valid_ds,
            epochs=epochs,
            # Shuffling in dataset
            shuffle=False,
            callbacks=self.full_model_callbacks,
            verbose=0
        )

        history_data = {
            'history': numpy_dict_to_json_format(self.full_model_history.history),
            'model_name': model_name
        }
        if self.use_test_split:
            test_scores = self.full_model.evaluate(self.test_ds)
            test_scores = dict(zip(self.full_model.metrics_names, test_scores))
            history_data['test_scores'] =  numpy_dict_to_json_format(test_scores)
        dump_json(history_data, os.path.join('history', model_name))

        if save_last_model:
            self.full_model.save(model_path + '_final', include_optimizer=False)

    def train_top_model_from_config(self):
        self.train_top_model(
            epochs=self.tl_epochs,
            optimizer_name=self.tl_optimizer_name,
            optimizer_lr=self.tl_optimizer_lr,
            early_stop_cb=self.tl_early_stop_cb,
            reduce_lr_cb=self.tl_reduce_lr_cb,
            tboard_cb=self.tl_tboard_cb,
            save_last_model=self.tl_save_last_model,
            jupyter_mode=self.tl_save_last_model
        )

    def train_full_model_from_config(self):
        self.train_full_model(
            epochs=self.ftune_epochs,
            optimizer_name=self.ftune_optimizer_name,
            optimizer_lr=self.ftune_optimizer_lr,
            weights_decay=self.ftune_weights_decay,
            early_stop_cb=self.ftune_early_stop_cb,
            reduce_lr_cb=self.ftune_reduce_lr_cb,
            tboard_cb=self.ftune_tboard_cb,
            save_intermediate_models=self.ftune_save_intermediate_models,
            save_last_model=self.ftune_save_last_model,
            jupyter_mode=self.ftune_save_last_model
        )
