import time
import json
import sys
import os
import abc
import copy
import numpy as np
import pandas as pd
from functools import partial

import keras
import keras.backend as k
from keras.models import Model, load_model, model_from_yaml
from keras.layers import Input, Concatenate, Conv2D, Lambda, Dense, Add, Average, Multiply
from keras.engine.training_utils import is_sequence, iter_sequence_infinite, should_run_validation
from keras.utils.data_utils import Sequence, OrderedEnqueuer, GeneratorEnqueuer
from keras.utils.generic_utils import Progbar, to_list, unpack_singleton
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
import keras.callbacks as cbks
import tensorflow as tf

from . import loss_and_metric
from .tensorboard_utils import *
from .ops import *
# from .data_utils import *

#####################################################################################################################
# GAN
#####################################################################################################################
# Base WAE-GAN Network
#####################################################################################################################
class WAE_GAN():
    def __init__(self, log, path_info, network_info, n_label, is_profiling=False):
        self.log = log
        self.path_info = path_info
        self.network_info = network_info
        self.n_label = n_label
        self.fixed_noise = np.array([None])
        self.metrics_names = ['main_loss', 'reconstruction', 'penalty_e',
                              'cluster_value','ssw', 'ssb', 'n_points', 'n_label',
                              'discriminator_loss',
                              'sharpness']
        self.TB = TensorBoardWrapper_GAN
        
        self.model_save_dir = self.path_info['model_info']['model_dir']
        self.train_models = None
        self.best_model_save = None
        
        self.e_sd = float(network_info['model_info']['e_sd'])
        self.lambda_e = float(network_info['model_info']['lambda_e'])
        
        if tf.__version__.startswith('2'):
            self.number_of_gpu = tf.config.experimental.get_visible_devices(device_type='GPU')
        else: self.number_of_gpu = len(k.tensorflow_backend._get_available_gpus())
        
        # profiling
        self.is_profiling = is_profiling
        if self.is_profiling:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata= tf.RunMetadata()
        else:
            self.run_options = None
            self.run_metadata = None
        
        
    def _load_model(self, model_yaml_path, custom_objects=None, verbose=0):
        yaml_file = open(model_yaml_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        if custom_objects is None: custom_objects = {}
        custom_objects['SelfAttention2D'] = SelfAttention2D
        model = model_from_yaml(loaded_model_yaml, custom_objects=custom_objects)
        if verbose:
            self.log.info(model_yaml_path.split('/')[-1])
            model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
        return model
    
    def _get_cluster_information_by_class_index(self, args):
        import tensorflow as tf
        latent_variable, class_info = args
        centroids_spread, centroids, unique_class, class_count, n_l = get_class_mean_by_class_index(latent_variable, class_info)
        n_points_mean = tf.reduce_mean(class_count)
        ssw = tf.reduce_sum(tf.math.square(latent_variable - centroids_spread))
        total_mean = tf.reduce_mean(latent_variable, axis=0)
        ssb = tf.reduce_sum(tf.math.square(centroids_spread - total_mean))
        return [ssw, ssb, n_points_mean, n_l]
    
    def build_model(self, model_yaml_dir=None, verbose=0):
        """
        verbose
            0: Not show any model
            1: Show AE, Discriminator model
            2: Show all models
        """
        # Load Models : encoder, decoder, discriminator
        if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, self.path_info['model_info']['model_architecture'])
        self.encoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder'], verbose=verbose==2)
        self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], verbose=verbose==2)
        self.discriminator_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], verbose=verbose==2)
        
        # build blocks
        self.image_shape = self.encoder_model.input_shape[1:]
        self.z_dim = self.encoder_model.output_shape[-1]
        
        real_image = Input(shape=self.image_shape, name='real_image', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')

        prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')
        latent_input = Input(shape=(self.z_dim,), name='latent_input', dtype='float32')

        fake_latent = self.encoder_model(real_image)
        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
        
        # GAN model for distance
        p_z = self.discriminator_model([prior_latent])
        q_z = self.discriminator_model([fake_latent])
        output = Concatenate(name='mlp_concat')([p_z, q_z])
        self.gan_model = Model(inputs=[real_image, prior_latent], outputs=[output], name='GAN_model')
        
        # Main model for WAE
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        latent_penalty = Lambda(get_qz_trick_loss, name='latent_penalty')(q_z)
        self.main_model = Model(inputs=[real_image], outputs=[recon_error, latent_penalty], name='Main_model')
        
        # Blur information
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        if verbose==2:
            self.log.info('Generative sample blurr model')
            self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            
        # cluster information
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([fake_latent, cls_info])
        self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        if verbose==2:
            self.log.info('Cluster information model')
            self.cluster_info_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            
        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu) #, cpu_relocation=True)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu) #, cpu_relocation=True)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        
        self.train_models_lr = {'discriminator':{'lr':float(self.network_info['model_info']['lr_e_adv']),
                                                 'decay':float(self.network_info['model_info']['lr_e_adv_decay'])},
                                'main':{'lr':float(self.network_info['model_info']['lr_e']),
                                        'decay':float(self.network_info['model_info']['lr_e_decay'])}}
            
        # Pretrain model
#         if self.pretrain: self.pretrain_encoder_model = Model(real_image, fake_latent, name='pretrain_encoder_model')
        
        if verbose:
            self.log.info('Main model')
            self.main_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            self.log.info('Discriminator model')
            self.gan_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
#             if self.pretrain & verbose==2: 
#                 self.log.info('Pretrain model')
#                 self.pretrain_encoder_model.summary(line_length=200, print_fn=self.log.info)
#                 sys.stdout.flush()
            
    def model_compile(self, verbose=0):
        self.log.info('Start models compile.')
        if self.network_info['model_info']['optimizer'] =='adam':
            optimizer_e = getattr(keras.optimizers,
                                  self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['main']['lr'],
                                                                                beta_1=float(self.network_info['model_info']['lr_e_beta1']))
            optimizer_e_adv = getattr(keras.optimizers,
                                      self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['discriminator']['lr'],
                                                                                    beta_1=float(self.network_info['model_info']['lr_e_adv_beta1']))
        else: 
            optimizer_e = getattr(keras.optimizers,
                                  self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['main']['lr'])
            optimizer_e_adv = getattr(keras.optimizers,
                                      self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['discriminator']['lr'])
        
        # GAN model compile
        self.encoder_model.trainable = False
        self.decoder_model.trainable = False
        self.discriminator_model.trainable = True
        self.parallel_gan_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['discriminator_loss']),  
                                        optimizer=optimizer_e_adv, options=self.run_options, run_metadata=self.run_metadata)
        
        # WAE model compile
        self.encoder_model.trainable = True
        self.decoder_model.trainable = True
        self.discriminator_model.trainable = False
        self.parallel_main_model.compile(loss={'mean_recon_error':getattr(loss_and_metric, self.network_info['model_info']['main_loss']), 
                                               'latent_penalty':getattr(loss_and_metric, self.network_info['model_info']['penalty_e']),
                                              },
                                             loss_weights=[1., self.lambda_e],
                                             optimizer=optimizer_e, options=self.run_options, run_metadata=self.run_metadata)
        
#         if self.pretrain:
#             self.encoder_model.trainable = True
#             self.pretrain_encoder_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['pretrain_loss']),
#                                                 optimizer=optimizer)
        if verbose:
            for name, model in self.train_models.items():
                self.log.info('%s model' % name)
                model.summary(line_length=200, print_fn=self.log.info)
                sys.stdout.flush()
            self.log.info('Model compile done.')
        
    def get_callbacks(self):
        ## Callback
        if 'callbacks' in self.network_info['training_info']:
            callbacks = [cb.strip() for cb in self.network_info['training_info']['callbacks'].split(',')]
            for idx, callback in enumerate(callbacks):
                if 'EarlyStopping' in callback:
                    callbacks[idx] = getattr(cbks, callback)(monitor=self.network_info['training_info']['monitor'],
                                                             mode=self.network_info['training_info']['mode'],
                                                             patience=int(self.network_info['training_info']['patience']),
                                                             min_delta=float(self.network_info['training_info']['min_delta']),
                                                             verbose=1)
                elif 'ModelCheckpoint' in callback:
                    self.best_model_save = True
                    callbacks[idx] = getattr(cbks, callback)(filepath=self.model_save_dir,
                                                             monitor=self.network_info['training_info']['monitor'],
                                                             mode=self.network_info['training_info']['mode'],
                                                             save_best_only=True, save_weights_only=False,
                                                             verbose=0)
                else:
                    callbacks[idx] = getattr(cbks, callback)()
        else:
            callbacks = []
        if 'None' not in self.network_info['tensorboard_info']['tensorboard_dir']: ## TODO : exist로 fix
                histogram_freq=int(self.network_info['tensorboard_info']['histogram_freq'])
                callbacks.append(self.TB(log_dir='%s' % (self.network_info['tensorboard_info']['tensorboard_dir']),
                                         histogram_freq=histogram_freq,
                                         batch_size=int(self.network_info['validation_info']['batch_size']),
                                         write_graph=self.network_info['tensorboard_info']['write_graph']=='True',
                                         write_grads=self.network_info['tensorboard_info']['write_grads']=='True',
                                         write_images=self.network_info['tensorboard_info']['write_images']=='True',
                                         write_weights_histogram=self.network_info['tensorboard_info']['write_weights_histogram']=='True', 
                                         write_weights_images=self.network_info['tensorboard_info']['write_weights_images']=='True',
                                         embeddings_freq=int(self.network_info['tensorboard_info']['embeddings_freq']),
                                         embeddings_metadata='metadata.tsv',
                                         tb_data_steps=1,
                                         tb_data_batch_size = 100))
        return callbacks
    
    def reset_weights(self, model):
        session = k.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'): 
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session) 
    
    def get_normalize_sym(self):
        if 'sigmoid' in self.decoder_model.layers[-1].get_config()['activation']: normalize_sym=False
        else: normalize_sym=True
        return normalize_sym
    
    def get_z_dim(self):
        return self.z_dim
    
    def get_n_label(self):
        return self.n_label
    
    def noise_sampler(self, size, z_dim, scale = 1., dist='gaussian'):
        """
        'gaussian': gaussian
        'spherical_uniform': spherical uniform
        """
        mean = np.zeros(z_dim)
        cov = scale**2.*np.identity(z_dim)
        noise = np.random.multivariate_normal(mean, cov, size).astype(np.float32)
        if dist=='spherical_uniform':
            noise = noise / np.sqrt(np.sum(noise * noise, axis=1))[:, np.newaxis]
        return noise

    def discriminator_sampler(self, x, y):
        noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
        return [x, noise], [np.zeros([x.shape[0],2], dtype='float32')]
    
    def main_sampler(self, x, y):
#         return [x], [x, np.zeros(x.shape[0], dtype='float32')]
        return [x], [np.zeros(x.shape[0], dtype='float32')] * 2

    def get_weights(self):
        weights = {}
        for name, model in self.train_models.items():
            weights[name] = model.get_weights()
        return weights
            
    def set_weights(self, weights):
        for name, model in self.train_models.items():
            model.set_weights(weights[name])
    
    def save(self, filepath, is_compile=True, overwrite=True, include_optimizer=True):
        # TODO: fix
        model_path = self.path_info['model_info']['weight']
        for name, model in self.train_models.items():
            model.save("%s/%s_%s" % (filepath, name, model_path), overwrite=overwrite, include_optimizer=include_optimizer)
        self.log.debug('Save model at %s' % filepath)
        if is_compile: self.model_compile()
        
    def load(self, filepath):
        # TODO: fix
        model_path = self.path_info['model_info']['weight']
        
        loss_list = [self.network_info['model_info']['main_loss'],
                     self.network_info['model_info']['penalty_e'],
                     self.network_info['model_info']['discriminator_loss']]
        load_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in loss_list])
#         load_dict['k'] = k
#         load_dict['tf'] = tf
        load_dict['SelfAttention2D'] = SelfAttention2D
        load_dict['get_qz_trick_loss'] = get_qz_trick_loss
        load_dict['get_qz_trick_with_weight_loss'] = get_qz_trick_with_weight_loss
        load_dict['mmd_penalty'] = mmd_penalty
        load_dict['get_b'] = get_b
        load_dict['get_b_estimation_var'] = get_b_estimation_var
        load_dict['get_b_penalty_loss'] = get_b_penalty_loss
        load_dict['mean_reconstruction_l2sq_loss_b'] = mean_reconstruction_l2sq_loss_b
        load_dict['mean_reconstruction_l2sq_loss_e'] = mean_reconstruction_l2sq_loss_e
        load_dict['get_class_mean_by_class_index'] = get_class_mean_by_class_index
        
        self.main_model = load_model("%s/%s_%s" % (filepath, "main", model_path), custom_objects=load_dict)
        self.gan_model = load_model("%s/%s_%s" % (filepath, "discriminator", model_path), custom_objects=load_dict)
        
        for layer in self.main_model.layers:
            if 'encoder' in layer.name: self.encoder_model = layer
            if 'decoder' in layer.name: self.decoder_model = layer
            if 'discriminator' in layer.name: self.discriminator_model = layer
        real_image = Input(shape=self.image_shape, name='real_image', dtype='float32')
        prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')
        latent_input = Input(shape=(self.z_dim,), name='latent_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')

        fake_latent = self.encoder_model(real_image)
        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image], outputs=[recon_image], name='ae_model')

        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([fake_latent, cls_info])
        self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        
         
        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu) #, cpu_relocation=True)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu) #, cpu_relocation=True)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        
        self.train_models_lr = {'discriminator':{'lr':float(self.network_info['model_info']['lr_e_adv']),
                                                 'decay':float(self.network_info['model_info']['lr_e_adv_decay'])},
                                'main':{'lr':float(self.network_info['model_info']['lr_e']),
                                        'decay':float(self.network_info['model_info']['lr_e_decay'])}}
        
        self.model_compile()
        self.log.info('Main model')
        self.main_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        self.log.info('Discriminator model')
        self.gan_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()        
                
    def save_weights(self, filepath, overwrite=True):
        # save weight only
        model_path = self.path_info['model_info']['weight']
        for name, model in self.train_models.items():
            model.save_weight("%s/%s_%spy" % (filepath, name, model_path), overwrite=overwrite)
    
    def load_weights(self, filepath, by_name=False,
                     skip_mismatch=False, reshape=False):
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        model_path = self.path_info['model_info']['weight']
        for name, model in self.train_models.items():
            filepath = "%s/%s_%spy" % (filepath, name, model_path)
            with h5py.File(filepath, mode='r') as f:
                if 'layer_names' not in f.attrs and 'model_weights' in f:
                    f = f['model_weights']
                if by_name:
                    saving.load_weights_from_hdf5_group_by_name(
                        f, model.layers, skip_mismatch=skip_mismatch,
                        reshape=reshape)
                else:
                    saving.load_weights_from_hdf5_group(
                        f, model.layers, reshape=reshape)
            
    def save_history(self, epoch=None, verbose=True):
        hist_path = os.path.join(self.model_save_dir, self.path_info['model_info']['history'])
        hist = self.history.history
        try: hist['epochs'].append(epoch)
        except: hist['epochs'] = [epoch]
        try:
            with open(hist_path, 'w+') as f:
                json.dump(hist, f)
        except:
            with open(hist_path, 'w+') as f:
                hist = dict([(ky, np.array(val).astype(np.float).tolist()) for (ky, val) in hist.items()])
                json.dump(hist, f)
        if verbose: self.log.info('Save history at %s' % hist_path)  
    
    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        dx, dy = self.discriminator_sampler(x, y)
        main_outs = self.parallel_main_model.train_on_batch(wx, wy, sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch([wx[0],y])
        if (self.fixed_noise != None).any():
#             gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0])
            gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0]//(4*self.number_of_gpu))
        else: raise ValueError('No fixed noise')
        d_outs = self.parallel_gan_model.train_on_batch(dx, dy, sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return main_outs + [cluster_value, ssw, ssb, n_points_mean, n_l] + [d_outs] + [np.min(gen_blurr)]
    
    def test_on_batch(self, x, y, sample_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        dx, dy = self.discriminator_sampler(x, y)
        main_outs = self.parallel_main_model.test_on_batch(wx, wy, sample_weight=sample_weight, reset_metrics = reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch([wx[0],y])
        if (self.fixed_noise != None).any():
            gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0]//(4*self.number_of_gpu))
        else: raise ValueError('No fixed noise')
        d_outs = self.parallel_gan_model.test_on_batch(dx, dy, sample_weight=sample_weight, reset_metrics = reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return main_outs + [cluster_value, ssw, ssb, n_points_mean, n_l] + [d_outs] + [np.min(gen_blurr)]
        
    def on_train_begin(self, x):
        real_image_blurriness = self.blurr_model.predict_on_batch(x)
        self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))
        self.fixed_noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
        
#     def pretrain_fit(self, generator):
#         #### TODO....
#         self.log.info("Pretrain start")
#         nprint = 20
#         nsteps = 500 # TODO, dinamix max step (sample size)
#         hist = []
#         for step in range(nsteps):
#             x, y = generator[step]
#             # dynamic sampler...
#             # traininb prograssbars...
#             noise = self.noise_sampler(x.shape[0], self.z_dim, self.noise_d)
#             raise Error
#             pretrain_outs = self.pretrain_encoder_model.train_on_batch(x, noise)
#             hist.append(pretrain_outs)
#             if step // nprint * nprint == step:
#                 print('%03d/%03d [' % (step, nsteps) + '='*(step//nprint+1)+'>'+'.'*(nsteps//nprint-step//nprint)+'] - loss %.5f' % hist[-1], end='\r')
#         print('%03d/%03d [' % (step+1, nsteps) +'='*(step//nprint+1)+'] - loss %.5f' % hist[-1])
#         self.log.info("Pretrain end")
        
    def fit_generator(self, generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      validation_data=None,
                      validation_steps=None,
                      validation_freq=1,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0,
                      warm_start=False,
                      warm_start_model=None,
                      save_frequency=None):
        """Modified fit_generator from keras.training_generator.training_generator 
        """
        
        val_workers = 0
        val_multiprocessing = False
        
        if warm_start:
            with open('./%s/hist.json' % (warm_start_model), 'r') as f:
                history = json.load(f)
            try:
                trained_epoch = int(history['epochs'][-1])
                if np.isnan(trained_epoch):
                    trained_epoch = int(history['epochs'][-2])
            except:
                trained_epoch = len(list(history.values())[0])
            epochs += trained_epoch
            epoch = initial_epoch+trained_epoch
            self.load(warm_start_model)
            self.log.info('Load %d epoch trained weights from %s' % (trained_epoch, warm_start_model))
        else:
            epoch = initial_epoch

        do_validation = bool(validation_data)
        for model in self.parallel_train_models.values():
            model._make_train_function()
        if do_validation:
            for model in self.parallel_train_models.values():
                model._make_test_function()

        use_sequence_api = is_sequence(generator)
        if not use_sequence_api and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the `keras.utils.Sequence'
                            ' class.'))

        # if generator is instance of Sequence and steps_per_epoch are not provided -
        # recompute steps_per_epoch after each epoch
        recompute_steps_per_epoch = use_sequence_api and steps_per_epoch is None

        if steps_per_epoch is None:
            if use_sequence_api:
                steps_per_epoch = len(generator)
            else:
                raise ValueError('`steps_per_epoch=None` is only valid for a'
                                 ' generator based on the '
                                 '`keras.utils.Sequence`'
                                 ' class. Please specify `steps_per_epoch` '
                                 'or use the `keras.utils.Sequence` class.')

        val_use_sequence_api = is_sequence(validation_data)
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__') or
                   val_use_sequence_api)
        if (val_gen and not val_use_sequence_api and
                not validation_steps):
            raise ValueError('`validation_steps=None` is only valid for a'
                             ' generator based on the `keras.utils.Sequence`'
                             ' class. Please specify `validation_steps` or use'
                             ' the `keras.utils.Sequence` class.')

        # Prepare display labels.
        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        callbacks = self.get_callbacks()
        
        self.history = cbks.History()
        _callbacks = [cbks.BaseLogger(
            stateful_metrics=self.metrics_names[1:])]
        if verbose:
            _callbacks.append(
                cbks.ProgbarLogger(
                    count_mode='steps',
                    stateful_metrics=self.metrics_names[1:]))
        _callbacks += (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(_callbacks)

        # TODO
# #         it's possible to callback a different model than self:
#         callback_model = model._get_callback_model()

        callback_model = self
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        
        self.tb_data = None
        
        ##############################################################
        # On train begin
        callbacks._call_begin_hook('train')
        self.on_train_begin(self.get_reference_images(generator))
                
        # Pretrain model
#         if self.pretrain: self.pretrain_fit(generator)
        ##############################################################
        
        enqueuer = None
        val_enqueuer = None

        try:
            if do_validation:
                val_data = validation_data
#                 if val_gen and val_workers > 0:
#                     # Create an Enqueuer that can be reused
#                     val_data = validation_data
#                     if is_sequence(val_data):
#                         val_enqueuer = OrderedEnqueuer(
#                             val_data,
#                             use_multiprocessing=val_multiprocessing)
#                         validation_steps = validation_steps or len(val_data)
#                     else:
#                         val_enqueuer = GeneratorEnqueuer(
#                             val_data,
#                             use_multiprocessing=val_multiprocessing)
#                     val_enqueuer.start(workers=val_workers,
#                                        max_queue_size=max_queue_size)
#                     val_enqueuer_gen = val_enqueuer.get()
#                 elif val_gen:
#                     val_data = validation_data
#                     if is_sequence(val_data):
#                         print('isinfinite true')
#                         val_enqueuer_gen = iter_sequence_infinite(val_data)
#                         validation_steps = validation_steps or len(val_data)
#                     else:
#                         val_enqueuer_gen = val_data
#                 else:
#                      # Prepare data for validation
#                     if len(validation_data) == 2:
#                         val_x, val_y = validation_data
#                         val_sample_weight = None
#                     elif len(validation_data) == 3:
#                         val_x, val_y, val_sample_weight = validation_data
#                     else:
#                         raise ValueError('`validation_data` should be a tuple '
#                                          '`(val_x, val_y, val_sample_weight)` '
#                                          'or `(val_x, val_y)`. Found: ' +
#                                          str(validation_data))
#                     val_x, val_y, val_sample_weights = model._standardize_user_data(
#                         val_x, val_y, val_sample_weight)
#                     val_data = val_x + val_y + val_sample_weights
#                     if model.uses_learning_phase and not isinstance(K.learning_phase(),
#                                                                     int):
#                         val_data += [0.]
                for cbk in callbacks:
                    cbk.validation_data = val_data


            if workers > 0:
                if use_sequence_api:
                    enqueuer = OrderedEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing,
                        shuffle=shuffle)
                else:
                    enqueuer = GeneratorEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if use_sequence_api:
                    output_generator = iter_sequence_infinite(generator)
                else:
                    output_generator = generator

            callbacks.model.stop_training = False
            
            if self.parallel_train_models == None:
                raise ValueError('No compiled training models')
                
            # Construct epoch logs.
            epoch_logs = {}
            while epoch < epochs:
                for model in self.parallel_train_models.values():
                    model.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
#                     print('step start ---%d---' % steps_done)
                    generator_output = next(output_generator)
#                     print('generator done')
                    
                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if x is None or len(x) == 0:
                        # Handle data tensors support when no input given
                        # step-size = 1 for data tensors
                        batch_size = 1
                    elif isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    # build batch logs
                    
#                     print('generator %s' % str(x[0].shape))
                    
                    batch_logs = {'batch': batch_index, 'size': batch_size}
                    
                    callbacks.on_batch_begin(batch_index, batch_logs)
                    
#                     print('before out')
                    
                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight,
                                               reset_metrics=False)

                    outs = to_list(outs)
                    
#                     print(' outs[0]=%f'%outs[0])
                    
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o
                    
                    callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
#                     generator.on_batch_end()
                    
                    batch_index += 1
                    steps_done += 1
                    
#                     if workers > 0:
#                         enqueuer.join_end_of_epoch()
#                     print('step done ---%d---' % steps_done)
#                     sys.stdout.flush()
        
#                     ######################################################################################
                    if callback_model.stop_training:
                        break

                ######################################################################################
                # Epoch finished.
#                 print('before join')
#                 sys.stdout.flush()
                
                if workers > 0:
                    enqueuer.join_end_of_epoch()
                    
#                 print('after join')
#                 sys.stdout.flush()
        
                if (do_validation and should_run_validation(validation_freq, epoch)):
                    # Note that `callbacks` here is an instance of
                    # `keras.callbacks.CallbackList`
                    if val_gen:
                        val_outs = self.evaluate_generator(
#                                 val_enqueuer_gen,
                            val_data,
                            validation_steps,
                            callbacks=callbacks,
#                                 callbacks=None,
                            use_multiprocessing=False,
                            workers=0)
                    else:
                        val_outs = self.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            callbacks=callbacks,
                            verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o
                
                sys.stdout.flush()
                
                callback_data = val_data[0]
                for cbk in callbacks:
                    cbk.validation_data = [callback_data]
                callbacks.on_epoch_end(epoch, epoch_logs)
                
                self.on_epoch_end(epoch)
                
                if callback_model.stop_training:
                    break
                
                epoch += 1
                
                if use_sequence_api and workers == 0:
                    generator.on_epoch_end()
                if val_gen: val_data.on_epoch_end()

                if recompute_steps_per_epoch:
#                     if workers > 0:
#                         enqueuer.join_end_of_epoch()

                    # recomute steps per epochs in case if Sequence changes it's length
                    steps_per_epoch = len(generator)

                    # update callbacks to make sure params are valid each epoch
                    callbacks.set_params({
                        'epochs': epochs,
                        'steps': steps_per_epoch,
                        'verbose': verbose,
                        'do_validation': do_validation,
                        'metrics': callback_metrics,
                    })
                
                if save_frequency is not None:
                    if epoch // save_frequency * save_frequency == epoch:
                        self.save(filepath=self.model_save_dir, is_compile=False)
                        self.save_history(epoch = epoch, verbose=False)
                ######################################################################################
        finally:
            try:
                if enqueuer is not None:
                    enqueuer.stop()
            finally:
                if val_enqueuer is not None:
                    val_enqueuer.stop()
#                     tb_enqueuer_gen.stop()

        callbacks._call_end_hook('train')
        
        if self.best_model_save: self.load(self.model_save_dir)
        else: self.save(filepath=self.model_save_dir, is_compile=False)
        return self.history
        
    def evaluate_generator(self, generator,
                           steps=None,
                           callbacks=None,
                           max_queue_size=10,
                           workers=0,
                           use_multiprocessing=False,
                           verbose=0):
        """Modified evaluate_generator from keras.training_generator.evaluate_generator"""
        
        for model in self.parallel_train_models.values():
            model._make_test_function()
            model.reset_metrics()
            
        steps_done = 0
        outs_per_batch = []
        batch_sizes = []
#         self.tb_data = []
        use_sequence_api = is_sequence(generator)
        if not use_sequence_api and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the `keras.utils.Sequence'
                            ' class.'))
        if steps is None:
            if use_sequence_api:
                steps = len(generator)
            else:
                raise ValueError('`steps=None` is only valid for a generator'
                                 ' based on the `keras.utils.Sequence` class.'
                                 ' Please specify `steps` or use the'
                                 ' `keras.utils.Sequence` class.')
        enqueuer = None

        # Check if callbacks have not been already configured
        if not isinstance(callbacks, cbks.CallbackList):
            callbacks = cbks.CallbackList(callbacks)
            callback_model = model._get_callback_model()
            callbacks.set_model(callback_model)
            callback_metrics = list(model.metrics_names)
            callback_params = {
                'steps': steps,
                'verbose': verbose,
                'metrics': callback_metrics,
            }
            callbacks.set_params(callback_params)

        callbacks.model.stop_training = False
        callbacks._call_begin_hook('test')

        try:
            if workers > 0:
                if use_sequence_api:
                    enqueuer = OrderedEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing)
                else:
                    enqueuer = GeneratorEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if use_sequence_api:
                    output_generator = iter_sequence_infinite(generator)
                else:
                    output_generator = generator

            if verbose == 1:
                progbar = Progbar(target=steps)

            while steps_done < steps:
                generator_output = next(output_generator)
                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                if batch_size == 0:
                    raise ValueError('Received an empty batch. '
                                     'Batches should contain '
                                     'at least one item.')

                batch_logs = {'batch': steps_done, 'size': batch_size}
                callbacks._call_batch_hook('test', 'begin', steps_done, batch_logs)
                outs = self.test_on_batch(x, y,
                                           sample_weight=sample_weight,
                                           reset_metrics=False)
                outs = to_list(outs)
                outs_per_batch.append(outs)
                
                for l, o in zip(model.metrics_names, outs):
                    batch_logs[l] = o
                callbacks._call_batch_hook('test', 'end', steps_done, batch_logs)

                steps_done += 1
                batch_sizes.append(batch_size)

                if verbose == 1:
                    progbar.update(steps_done)
                
#                 self.tb_data.append([x, y])
            callbacks._call_end_hook('test')
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        averages = [float(outs_per_batch[-1][0])]  # index 0 = 'loss'
        for i in range(1, len(outs)):
            averages.append(np.float64(outs_per_batch[-1][i]))
        return unpack_singleton(averages)

    def get_reference_images(self, generator):
        return np.concatenate([generator[i][0] for i in range(4)])
    
    def _update_lr(self, epoch, lr, decay=0., lr_schedule="basic"):
        # see: https://github.com/keras-team/keras/issues/7874
        if lr_schedule == "basic":
            lr *= (1. / (1. + decay * epoch))
        elif lr_schedule == "manual":
            if epoch == 30:
                decay = decay / 2.
            if epoch == 50:
                decay = decay / 5.
            if epoch == 100:
                decay = decay / 10.
            lr *= decay
        elif lr_schedule == "manual_smooth":
            decay = decay / np.exp(np.log(100.) / epoch)
            lr *= decay
        else: 
            raise ValueError("No schedule information with %s" % lr_schedule)
        return lr
    
    def on_epoch_end(self, epoch): # epoch_logs
        for name in self.train_models_lr.keys():
            if self.train_models_lr[name]['decay'] > 0.:
                self.train_models_lr[name]['lr'] = self._update_lr(epoch, lr=self.train_models_lr[name]['lr'],
                                                                   decay=self.train_models_lr[name]['decay'])
                k.set_value(self.parallel_train_models[name].optimizer.lr, self.train_models_lr[name]['lr'])
   
 class WAE_MMD():
#     def __init__(self, log, path_info, network_info, n_label):
#         self.log = log
#         self.path_info = path_info
#         self.network_info = network_info
#         self.n_label = n_label
        
#         self.model_save_dir = self.path_info['model_info']['model_dir']
#         self.wae_lambda = float(network_info['model_info']['wae_lambda'])
#         self.decay = float(network_info['model_info']['decay'])
#         self.lr_schedule = self.network_info['model_info']['lr_schedule']
#         self.pretrain = network_info['model_info']['pretrain']=='True'
        
#         self.fixed_noise = np.array([None])
#         self.noise_d = float(network_info['model_info']['noise_d'])
#         self.metrics_names = ['wae_loss', 'wae_reconstruction', 'wae_penalty', 'log_wae_penalty', 
#                               'cluster_value','ssw', 'ssb', 'msw', 'msb', 'n_points', 'n_label',
#                               'sharpness',
#                               # 'fake_mean', 'fake_var'
#                              ]
        
#         self.best_model_save = None
#         self.number_of_gpu = len(k.tensorflow_backend._get_available_gpus())
        
#         self.TB = TensorBoardWrapper_MMD
        
        
#     def _load_model(self, model_yaml_path, custom_objects=None, verbose=0):
#         # load YAML and create model
#         yaml_file = open(model_yaml_path, 'r')
#         loaded_model_yaml = yaml_file.read()
#         yaml_file.close()
#         if custom_objects is not None:
#             custom_objects['Attention2D'] = Attention2D
#         else:
#             custom_objects = {'Attention2D':Attention2D}
#         model = model_from_yaml(loaded_model_yaml, custom_objects=custom_objects)
#         if verbose:
#             self.log.info(model_yaml_path.split('/')[-1])
#             model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
#         return model
    
#     def _init_laplace_filter(self, shape, dtype=None):
#         return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).reshape([3,3,1,1])
    
#     def _get_compute_blurriness(self, image_shape):
#         def set_rgb_to_gray(x):
#             import keras.backend as k
#             return tf.image.rgb_to_grayscale(x)
#         def get_variance(x):
#             import keras.backend as k
#             return k.var(x, axis=(1,2,3))
#         image_input = Input(shape=image_shape, name='image_input', dtype='float32')
#         if image_shape[-1] > 1: # RGB
#             image = Lambda(set_rgb_to_gray, name='rgb_to_gray')(image_input)
#         else: image=image_input
#         # laplace_transform = Conv2D(filters=1, kernel_size=3,
#         #                            kernel_initializer=self._init_laplace_filter, bias_initializer='zeros', padding='valid')(image)
#         laplace_transform = Conv2D(filters=1, kernel_size=3,
#                                    kernel_initializer=self._init_laplace_filter, bias_initializer='zeros', padding='same')(image) # TODO : Check
#         laplace_var = Lambda(get_variance, name='variance')(laplace_transform)
#         return Model(image_input, laplace_var, name='blurriness')

#     def _get_cluster_info(self, z_dim, n_label):
#         def get_class_points(args, z_dim, n_label):
#             import keras.backend as k
#             latent = args[0]
#             cls_info = args[1]
#             latent_T = k.transpose(latent)
#             wpts = k.stack([k.transpose(tf.multiply(latent_T, cls_info[:,i])) for i in range(n_label)], axis=1)
#             return wpts
#         def get_n_label(x):
#             import keras.backend as k
#             return tf.count_nonzero(x, dtype=tf.float32)
#         def get_n_points(x):
#             import keras.backend as k
#             return k.sum(x, axis=0)
#         def get_centroid(x): 
#             import keras.backend as k
#             return k.sum(x[0], axis=0)/(k.expand_dims(x[1], axis=-1)+1e-14)
#         def get_distance(x): 
#             import keras.backend as k
#             return k.sum(k.square(x[0]-k.expand_dims(x[2], axis=-1)*x[1]), axis=2)
#         def get_sw(x):
#             import keras.backend as k
#             return k.sum(x, axis=0)
#         def get_msw(x):
#             import keras.backend as k
#             return k.sum(x[0])/(k.sum(x[1])-x[2]+1e-14)
#         def get_tot_mean(x):
#             import keras.backend as k
#             return k.mean(x, axis=0)
#         def get_sb(x):
#             import keras.backend as k
#             return k.sum(k.square(x[0]-x[1]), axis=1)
#         def get_msb(x):
#             import keras.backend as k
#             return k.sum(x[0])/(x[1]-1.)
#         # def get_msb(x, n_label):
#         #     import keras.backend as k
#         #     return k.stack([k.sum(k.square(x-x[i,:]))/(n_label-1.) for i in range(n_label)])
            
#         latent_input = Input(shape=(z_dim,), name='latent_input', dtype='float32')
#         cls_input = Input(shape=(n_label,), name='class_info_input', dtype='float32')

#         wpts = Lambda(get_class_points, arguments={"z_dim":z_dim, "n_label":n_label}, 
#                       name='get_within_class_pts')([latent_input, cls_input])
#         n_points = Lambda(get_n_points, name='n_points')(cls_input)
#         n_l = Lambda(get_n_label, name='n_label')(n_points)
#         centroid = Lambda(get_centroid, name='centroid')([wpts, n_points])
#         distance = Lambda(get_distance, name='distance')([wpts, centroid, cls_input])
#         sw = Lambda(get_sw, name='get_sw')([distance])
#         msw = Lambda(get_msw, name='get_msw')([sw, n_points, n_l])
#         tot_mean = Lambda(get_tot_mean, name='global_centroid')(latent_input)
#         sb = Lambda(get_sb, name='get_sb')([centroid, tot_mean])
#         msb = Lambda(get_msb, name='get_msb')([sb, n_l])
#         return Model([latent_input, cls_input], [sw, sb, msw, msb, n_points, n_l], name='cluster_info')
        
#     def _get_mmd_penalty_loss(self, arg, sigma=10., zdim=8, kernel='RBF', p_z='normal'):
#         import keras.backend as k
#         from utils import mmd_penalty
#         sample_pz = arg[0]
#         sample_qz = arg[1]
#         nsize = k.sum(k.ones_like(sample_pz[:,0]))
#         stat = mmd_penalty(sample_pz, sample_qz, sigma, nsize, kernel, p_z, zdim)
#         return stat + k.zeros_like(sample_pz[:,0])
    
# #     # TODO : class 별 mmd...
# #     def _get_mmd_with_cls_weight(self, arg, sigma=10., zdim=8, kernel='RBF', p_z='normal', n_label=10):
# #         import keras.backend as k
# #         from utils import mmd_penalty
# #         sample_pz = arg[0]
# #         sample_qz = arg[1]
# #         cls_info = arg[2]
# #         stat = k.stack([mmd_penalty(tf.boolean_mask(sample_pz,tf.greater(k.ones_like(sample_pz)*cls_info[:,i],0)),
# #                                     tf.boolean_mask(sample_qz,tf.greater(k.ones_like(sample_qz)*cls_info[:,i],0)),
# #                                     sigma, tf.reduce_sum(cls_info[:,i]), kernel, p_z, zdim) for i in range(n_label)], axis=0)
# #         return tf.reduce_mean(stat) + k.zeros_like(sample_pz[:,0])
        
#     def build_model(self, model_yaml_dir=None, verbose=0):
#         """
#         verbose
#             0: Not show any model
#             1: Show AE, Discriminator model
#             2: Show all models
#         """
#         # Load Models : encoder, decoder, discriminator
#         if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, self.path_info['model_info']['model_architecture'])
        
#         self.encoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder'], verbose=verbose==2)
#         self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], verbose=verbose==2)
       
#         # build blocks
#         self.image_shape = self.encoder_model.input_shape[1:]
#         self.z_dim = self.encoder_model.output_shape[-1]
        
#         real_image = Input(shape=self.image_shape, name='real_image', dtype='float32')
#         cls_info = Input(shape=(self.n_label,), name='class_info_input', dtype='float32')

#         prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')

#         fake_latent = self.encoder_model(real_image)
#         recon_image = self.decoder_model(fake_latent)
#         self.ae_model = Model(inputs=[real_image], outputs=[recon_image], name='ae_model')
#         if verbose==2:
#             self.log.info('Auto-Encoder model')
#             self.ae_model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
        
#         self.blurr_model = self._get_compute_blurriness(self.image_shape)
#         gen_image = self.decoder_model(prior_latent)
#         gen_sharpness = self.blurr_model(gen_image)
#         self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
#         if verbose==2:
#             self.log.info('Generative sample blurr model')
#             self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
            
#         # cluster information
#         self.get_cluster_info = self._get_cluster_info(self.z_dim, self.n_label)
#         sw, sb, msw, msb, n_point, n_l = self.get_cluster_info([fake_latent, cls_info])
#         self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[sw, sb, msw, msb, n_point, n_l], name='get_cluster_info')
#         if verbose==2:
#             self.log.info('Cluster information model')
#             self.cluster_info_model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
        
#         # WAE model
#         latent_penalty = Lambda(self._get_mmd_penalty_loss, name='latent_penalty',
#                                 arguments={'sigma':1.,
#                                            'zdim':self.z_dim,
#                                            'kernel':'IMQ', 'p_z':'normal'})([prior_latent, fake_latent])
#         self.wae_model = Model(inputs=[real_image, cls_info, prior_latent], outputs=[recon_image, latent_penalty], name='WAE_model')
        
#         try:
# #             self.parallel_wae_model = multi_gpu_model(self.wae_model, gpus=number_of_gpu, cpu_relocation=True)
#             self.parallel_wae_model = multi_gpu_model(self.wae_model, gpus=self.number_of_gpu)
#             self.log.info("Training using multiple GPUs")
#         except ValueError:
#             self.parallel_wae_model = self.wae_model
#             self.log.info("Training using single GPU or CPU")
        
#         # Pretrain model
#         if self.pretrain: self.pretrain_encoder_model = Model(real_image, fake_latent, name='pretrain_encoder_model')
#         if verbose:
#             self.log.info('WAE model')
#             self.wae_model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
#             if self.pretrain & verbose==2: 
#                 self.log.info('Pretrain model')
#                 self.pretrain_encoder_model.summary(line_length=200, print_fn=self.log.info)
#                 sys.stdout.flush()
            
#     def model_compile(self, verbose=0):
#         self.log.info('Start models compile.')
        
#         if self.network_info['model_info']['optimizer'] =='adam':
#             optimizer = getattr(keras.optimizers,
#                                 self.network_info['model_info']['optimizer'])(lr=float(self.network_info['model_info']['lr']),
#                                                                               beta_1=float(self.network_info['model_info']['beta1']))
#         else: optimizer = getattr(keras.optimizers,
#                                   self.network_info['model_info']['optimizer'])(lr=float(self.network_info['model_info']['lr']))
#         self.parallel_wae_model.compile(loss={'decoder':getattr(loss_and_metric, self.network_info['model_info']['wae_loss']),
#                                      'latent_penalty':getattr(loss_and_metric, self.network_info['model_info']['wae_penalty']),
#                                      },
#                                loss_weights=[1., self.wae_lambda],
#                                optimizer=optimizer)
#         if verbose:
#             self.log.info('WAE model')
#             self.parallel_wae_model.summary(line_length=200, print_fn=self.log.info)
#             sys.stdout.flush()
        
#         if self.pretrain:
#             self.pretrain_encoder_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['pretrain_loss']),
#                                                 optimizer=optimizer)
#             if verbose:
#                 self.log.info('Pretrain model')
#                 self.pretrain_encoder_model.summary(line_length=200, print_fn=self.log.info)
#                 sys.stdout.flush()
#         self.log.info('Model compile done.')
        
#     def get_callbacks(self, validation_data=None):
#         ## Callback
#         if 'callbacks' in self.network_info['training_info']:
#             callbacks = [cb.strip() for cb in self.network_info['training_info']['callbacks'].split(',')]
#             for idx, callback in enumerate(callbacks):
#                 if 'EarlyStopping' in callback:
#                     callbacks[idx] = getattr(cbks, callback)(monitor=self.network_info['training_info']['monitor'],
#                                                              mode=self.network_info['training_info']['mode'],
#                                                              patience=int(self.network_info['training_info']['patience']),
#                                                              min_delta=float(self.network_info['training_info']['min_delta']),
#                                                              verbose=1)
#                 elif 'ModelCheckpoint' in callback:
#                     self.best_model_save = True
#                     #self.log.debug('monitor:%s',model_path)
#                     callbacks[idx] = getattr(cbks, callback)(filepath=self.model_save_dir,
#                                                              monitor=self.network_info['training_info']['monitor'],
#                                                              mode=self.network_info['training_info']['mode'],
#                                                              save_best_only=True, save_weights_only=False,
#                                                              verbose=0)
#                 else:
#                     callbacks[idx] = getattr(cbks, callback)()
#         else:
#             callbacks = []
#         if 'None' not in self.network_info['tensorboard_info']['tensorboard_dir']: ## TODO : exist로 fix
#                 histogram_freq=int(self.network_info['tensorboard_info']['histogram_freq'])
#                 callbacks.append(self.TB(log_dir='%s' % (self.network_info['tensorboard_info']['tensorboard_dir']),
#                                          histogram_freq=histogram_freq,
#                                          batch_size=int(self.network_info['validation_info']['batch_size']),
#                                          write_graph=self.network_info['tensorboard_info']['write_graph']=='True',
#                                          write_grads=self.network_info['tensorboard_info']['write_grads']=='True',
#                                          write_images=self.network_info['tensorboard_info']['write_images']=='True',
#                                          write_weights_histogram=self.network_info['tensorboard_info']['write_weights_histogram']=='True', 
#                                          write_weights_images=self.network_info['tensorboard_info']['write_weights_images']=='True',
#                                          embeddings_freq=int(self.network_info['tensorboard_info']['embeddings_freq']),
#                                          embeddings_metadata='metadata.tsv',
#                                          tb_data_steps=1,
#                                          n_class = self.n_label))
#         return callbacks
    
#     def get_normalize_sym(self):
#         if 'sig' in self.decoder_model.layers[-1].get_config()['activation']: normalize_sym=False
#         else: normalize_sym=True
#         return normalize_sym
    
#     def get_z_dim(self):
#         return self.z_dim
    
#     def get_n_label(self):
#         return self.n_label
    
#     def gaussian_noise_sampler(self, size, scale = 1., dist='gaussian'):
#         mean = np.zeros(self.z_dim)
#         cov = scale**2.*np.identity(self.z_dim)
#         noise = np.random.multivariate_normal(mean, cov, size).astype(np.float32)
#         ## sperical normal
#         if dist=='sperical':
#             noise = noise / np.sqrt(np.sum(noise * noise, axis=1))[:, np.newaxis]
#         return noise
    
#     def wae_sampler(self, x, y):
#         noise = self.gaussian_noise_sampler(x.shape[0], self.noise_d)
#         cls_info = to_categorical(y, num_classes=self.n_label)
#         return [x, cls_info, noise], [x, np.zeros(x.shape[0], dtype='float32')]

#     def get_weights(self):
#         return self.wae_model.get_weights()
            
#     def set_weights(self, weights):
#         self.wae_model.set_weights(weights)
    
#     def save(self, filepath, is_compile=True, overwrite=True, include_optimizer=True):
#         model_path = self.path_info['model_info']['weight']
#         for name, model in self.train_models.items():
#             model.save("%s/%s_%s" % (filepath, name, model_path), overwrite=overwrite, include_optimizer=include_optimizer)
#         self.log.debug('Save model at %s' % filepath)
#         if is_compile: self.model_compile()
        
    
#     def load(self, filepath):
#         model_path = self.path_info['model_info']['weight']
#         wae_loss = [self.network_info['model_info']['wae_loss'], 
#                     self.network_info['model_info']['wae_penalty'],
#                    ]
#         wae_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in wae_loss])
#         wae_dict['_get_mmd_penalty_loss'] = self._get_mmd_penalty_loss
#         wae_dict['Attention2D'] = Attention2D   
#         self.wae_model = load_model("%s/%s_%s" % (filepath, "wae", model_path), custom_objects=wae_dict)
        
#         for layer in self.wae_model.layers:
#             if 'encoder' in layer.name: self.encoder_model = layer
#             if 'decoder' in layer.name: self.decoder_model = layer
#         real_image = Input(shape=self.image_shape, name='real_image', dtype='float32')
#         prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')
#         cls_info = Input(shape=(self.n_label,), name='class_info_input', dtype='float32')

#         fake_latent = self.encoder_model(real_image)
#         recon_image = self.decoder_model(fake_latent)
#         self.ae_model = Model(inputs=[real_image], outputs=[recon_image], name='ae_model')

#         gen_image = self.decoder_model(prior_latent)
#         gen_sharpness = self.blurr_model(gen_image)
#         self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        
#         sw, sb, msw, msb, n_point, n_l = self.get_cluster_info([fake_latent, cls_info])
#         self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[sw, sb, msw, msb, n_point, n_l], name='get_cluster_info')
        
#         self.model_compile()
#         self.log.info('Loaded WAE model')
#         self.wae_model.summary(line_length=200, print_fn=self.log.info)
#         sys.stdout.flush()
#         self.log.info('Loaded Discriminator model: GAN')
#         self.gan_model.summary(line_length=200, print_fn=self.log.info)
#         sys.stdout.flush()
        
#     def save_weights(self, filepath, overwrite=True):
#         # save weight only
#         model_path = self.path_info['model_info']['weight']
#         self.wae_model.save_weight("%s/wae_%spy" % (filepath, model_path), overwrite=overwrite)
    
#     def load_weights(self, filepath, by_name=False,
#                      skip_mismatch=False, reshape=False):
#         if h5py is None:
#             raise ImportError('`load_weights` requires h5py.')
#         model_path = self.path_info['model_info']['weight']
#         model = self.wae_model
#         filepath = "%s/%s_%spy" % (filepath, 'wae', model_path)
#         with h5py.File(filepath, mode='r') as f:
#             if 'layer_names' not in f.attrs and 'model_weights' in f:
#                 f = f['model_weights']
#             if by_name:
#                 saving.load_weights_from_hdf5_group_by_name(
#                     f, model.layers, skip_mismatch=skip_mismatch,
#                     reshape=reshape)
#             else:
#                 saving.load_weights_from_hdf5_group(
#                     f, model.layers, reshape=reshape)
            
#     def save_history(self, epoch=None, verbose=True):
#         hist_path = os.path.join(self.model_save_dir, self.path_info['model_info']['history'])
#         hist = self.history.history
#         try: hist['epochs'].append(epoch)
#         except: hist['epochs'] = [epoch]
#         with open(hist_path, 'w') as f:
#             json.dump(hist, f) 
#         if verbose: self.log.info('Save history at %s' % hist_path)  
    
#     # def get_variance(self, fake_z, y):
#     #     sw = []
#     #     sb = []
#     #     tot_mean = np.mean(fake_z, axis=0)
#     #     for cls in np.unique(y):
#     #         wpts = fake_z[y==cls]
#     #         centroid = np.mean(wpts, axis=0)
#     #         distance = np.sum(np.square(wpts - centroid), axis=1)
#     #         sw.append(np.sum(distance))
#     #         sb.append(np.sum(np.square(np.mean(fake_z[y==cls], axis=0)-tot_mean)))
#     #         # self.log.info("%s (n %s): mw %s b %s" % (cls, wpts.shape[0],  sw[-1]/wpts.shape[0]*1., sb[-1]))
#     #     ssw = np.sum(sw)
#     #     ssb = np.sum(sb)
#     #     return ssw/(fake_z.shape[0]-np.unique(y).shape[0])*1., ssb/(np.unique(y).shape[0]-1.)
    
#     def train_on_batch(self, x, y):
#         wx, wy = self.wae_sampler(x, y)
#         wae_outs = self.parallel_wae_model.train_on_batch(wx, wy)    
#         # ######debug#############
#         # encoder = self.wae_model.get_layer('encoder')
#         # fake_latent_example = encoder.predict_on_batch(wx[:1])
#         # ######debug#############
#         sw, sb, msw, msb, n_point, n_l = self.cluster_info_model.predict_on_batch(wx[:2])
#         if (self.fixed_noise != None).any(): 
# #             gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0])
#             gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0]//(4*self.number_of_gpu))
#         else: raise ValueError('No fixed noise')
#         cluster_value = np.sum(sw)/np.sum(sb*n_point+1e-14)
#         return [wae_outs[0],wae_outs[1],wae_outs[2],np.log(max(0., wae_outs[2]))] \
#                 + [cluster_value, np.sum(sw), np.sum(sb), msw, msb, np.mean(n_point), n_l] \
#                 + [np.min(gen_blurr)] #+ [np.mean(fake_latent_example), np.var(fake_latent_example)]
    
#     def test_on_batch(self, x, y):
#         wx, wy = self.wae_sampler(x, y)
#         wae_outs = self.parallel_wae_model.test_on_batch(wx, wy)
#         # ######debug#############
#         # encoder = self.wae_model.get_layer('encoder')
#         # fake_latent_example = encoder.predict_on_batch(wx[:1])
#         # ######debug#############        
#         sw, sb, msw, msb, n_point, n_l = self.cluster_info_model.predict_on_batch(wx[:2])
#         if (self.fixed_noise != None).any(): 
# #             gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0])
#             gen_blurr = self.gen_blurr_model.predict(self.fixed_noise, batch_size=y.shape[0]//(4*self.number_of_gpu))
#         else: raise ValueError('No fixed noise')
#         cluster_value = np.sum(sw)/np.sum(sb*n_point+1e-14)
#         return [wae_outs[0],wae_outs[1],wae_outs[2],np.log(max(0., wae_outs[2]))] \
#                 + [cluster_value, np.sum(sw), np.sum(sb), msw, msb, np.mean(n_point), n_l] \
#                 + [np.min(gen_blurr)] #+ [np.mean(fake_latent_example), np.var(fake_latent_example)]
        
#     def on_train_begin(self, x):
#         real_image_blurriness = self.blurr_model.predict_on_batch(x)
#         self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))
#         self.fixed_noise = self.gaussian_noise_sampler(x.shape[0], self.noise_d)
#         np.random.seed()
        
#     def pretrain_fit(self, generator):
#         #### TODO....
#         self.log.info("Pretrain start")
#         nprint = 20
#         nsteps = 500 # TODO, dinamix max step (sample size)
#         hist = []
#         for step in range(nsteps):
#             x, y = generator[step]
#             # dynamic sampler...
#             # traininb prograssbars...
#             noise = self.gaussian_noise_sampler(x.shape[0], self.noise_d)
#             raise Error
#             pretrain_outs = self.pretrain_encoder_model.train_on_batch(x, noise)
#             hist.append(pretrain_outs)
#             if step // nprint * nprint == step:
#                 print('%03d/%03d [' % (step, nsteps) + '='*(step//nprint+1)+'>'+'.'*(nsteps//nprint-step//nprint)+'] - loss %.5f' % hist[-1], end='\r')
#         print('%03d/%03d [' % (step+1, nsteps) +'='*(step//nprint+1)+'] - loss %.5f' % hist[-1])
#         self.log.info("Pretrain end")
        
#     def fit_generator(self, generator,
#                       steps_per_epoch=None,
#                       epochs=1,
#                       verbose=1,
#                       validation_data=None,
#                       validation_steps=None,
#                       class_weight=None,
#                       max_queue_size=10,
#                       workers=1,
#                       use_multiprocessing=False,
#                       shuffle=True,
#                       initial_epoch=0,
#                       warm_start=False,
#                       warm_start_model=None,
#                       save_frequency=None):
#         """Modified fit_generator from keras.training_generator.training_generator
#         """
        
#         if warm_start:
#             with open('./%s/hist.json' % (warm_start_model), 'r') as f:
#                 history = json.load(f)
#             try: 
#                 if history['epochs'][-1] is not None: trained_epoch = history['epochs'][-1]
#                 else: trained_epoch = len(list(history.values())[0])
#             except:
#                 trained_epoch = len(list(history.values())[0])
#             epochs += trained_epoch
#             epoch = initial_epoch+trained_epoch
#             self.load(warm_start_model)
#             self.log.info('Load %d epoch trained weights from %s' % (trained_epoch, warm_start_model))
#         else:
#             epoch = initial_epoch
        
#         do_validation = bool(validation_data)
#         is_sequence = isinstance(generator, Sequence)
#         if not is_sequence and use_multiprocessing and workers > 1:
#             warnings.warn(
#                 UserWarning('Using a generator with `use_multiprocessing=True`'
#                             ' and multiple workers may duplicate your data.'
#                             ' Please consider using the`keras.utils.Sequence'
#                             ' class.'))
#         if steps_per_epoch is None:
#             if is_sequence:
#                 steps_per_epoch = len(generator)
#             else:
#                 raise ValueError('`steps_per_epoch=None` is only valid for a'
#                                  ' generator based on the '
#                                  '`keras.utils.Sequence`'
#                                  ' class. Please specify `steps_per_epoch` '
#                                  'or use the `keras.utils.Sequence` class.')

#         # python 2 has 'next', 3 has '__next__'
#         # avoid any explicit version checks
#         val_gen = (hasattr(validation_data, 'next') or
#                    hasattr(validation_data, '__next__') or
#                    isinstance(validation_data, Sequence))
#         if (val_gen and not isinstance(validation_data, Sequence) and
#                 not validation_steps):
#             raise ValueError('`validation_steps=None` is only valid for a'
#                              ' generator based on the `keras.utils.Sequence`'
#                              ' class. Please specify `validation_steps` or use'
#                              ' the `keras.utils.Sequence` class.')
#         if not (val_gen):
#             raise ValueError('only valid for generator validation input now...')

#         # Prepare display labels.
#         out_labels = self.metrics_names
#         callback_metrics = out_labels + ['val_' + n for n in out_labels]

#         # prepare callbacks
#         callbacks = self.get_callbacks(validation_data=validation_data)
#         self.history = cbks.History()
#         _callbacks = [cbks.BaseLogger(
#             stateful_metrics=self.metrics_names)]
#         if verbose:
#             _callbacks.append(
#                 cbks.ProgbarLogger(
#                     count_mode='steps',
#                     stateful_metrics=self.metrics_names))
#         _callbacks += (callbacks or []) + [self.history]
#         callbacks = cbks.CallbackList(_callbacks)

#         # TODO
# #         # it's possible to callback a different model than self:
# #         if hasattr(model, 'callback_model') and model.callback_model:
# #             callback_model = model.callback_model
# #         else:
# #             callback_model = model
#         callback_model = self
#         callbacks.set_model(callback_model)
#         callbacks.set_params({
#             'epochs': epochs,
#             'steps': steps_per_epoch,
#             'verbose': verbose,
#             'do_validation': do_validation,
#             'metrics': callback_metrics,
#         })
#         callbacks.on_train_begin()
        
#         ##############################################################
#         x = np.concatenate([generator[i][0] for i in range(4)])
#         self.on_train_begin(x)
        
#         if self.pretrain: self.pretrain_fit(generator)
        
#         enqueuer = None
#         val_enqueuer = None

#         try:
#             if do_validation:
#                 if val_gen: # and workers > 0: ## TODO : val_workers...
#                     # # Create an Enqueuer that can be reused
#                     # val_data = validation_data
#                     # if isinstance(val_data, Sequence):
#                     #     val_enqueuer = OrderedEnqueuer(
#                     #         val_data,
#                     #         use_multiprocessing=use_multiprocessing)
#                     #     validation_steps = validation_steps or len(val_data)
#                     # else:
#                     #     val_enqueuer = GeneratorEnqueuer(
#                     #         val_data,
#                     #         use_multiprocessing=use_multiprocessing)
#                     # val_enqueuer.start(workers=workers,
#                     #                    max_queue_size=max_queue_size)
#                     # val_enqueuer_gen = val_enqueuer.get()
#                 # #elif val_gen:
#                 # else:
#                     val_data = validation_data
#                     if isinstance(val_data, Sequence): # TODO
#                         val_enqueuer_gen = iter_sequence_infinite(val_data)
#                         validation_steps = validation_steps or len(val_data)
#                     else:
#                         val_enqueuer_gen = val_data
#                 # TODO
# #                 else:
# #                     # Prepare data for validation
# #                     if len(validation_data) == 2:
# #                         val_x, val_y = validation_data
# #                         val_sample_weight = None
# #                     elif len(validation_data) == 3:
# #                         val_x, val_y, val_sample_weight = validation_data
# #                     else:
# #                         raise ValueError('`validation_data` should be a tuple '
# #                                          '`(val_x, val_y, val_sample_weight)` '
# #                                          'or `(val_x, val_y)`. Found: ' +
# #                                          str(validation_data))
# #                     val_x, val_y, val_sample_weights = model._standardize_user_data(
# #                         val_x, val_y, val_sample_weight)
# #                     val_data = val_x + val_y + val_sample_weights
# #                     if model.uses_learning_phase and not isinstance(K.learning_phase(),
# #                                                                     int):
# #                         val_data += [0.]
#                 for cbk in callbacks:
#                     cbk.validation_data = val_data

#             # TODO
#             if workers > 0:
#                 if is_sequence: # TODO
#                     enqueuer = OrderedEnqueuer(
#                         generator,
#                         use_multiprocessing=use_multiprocessing,
#                         shuffle=shuffle)
#                 else:
#                     enqueuer = GeneratorEnqueuer(
#                         generator,
#                         use_multiprocessing=use_multiprocessing)
#                 enqueuer.start(workers=workers, max_queue_size=max_queue_size)
#                 output_generator = enqueuer.get()
#             else:
#                 if is_sequence:
#                     output_generator = iter_sequence_infinite(generator)
#                 else:
#                     output_generator = generator

#             callback_model.stop_training = False
                
#             # Construct epoch logs.
#             epoch_logs = {}
#             while epoch < epochs:
#                 model = self.wae_model
#                 for m in model.stateful_metric_functions:
#                     m.reset_states()
#                 callbacks.on_epoch_begin(epoch)
#                 steps_done = 0
#                 batch_index = 0
#                 while steps_done < steps_per_epoch:
#                     generator_output = next(output_generator)
                    
#                     ## TODO : sample weight
#                     if not hasattr(generator_output, '__len__'):
#                         raise ValueError('Output of generator should be '
#                                          '`(x, y)`. Found: ' +
#                                          str(generator_output))

#                     if len(generator_output) == 2:
#                         x, y = generator_output
#                     else:
#                         raise ValueError('Output of generator should be '
#                                          '`(x, y)`. Found: ' +
#                                          str(generator_output))
#                     # build batch logs
#                     batch_logs = {}
#                     if x is None or len(x) == 0:
#                         # Handle data tensors support when no input given
#                         # step-size = 1 for data tensors
#                         batch_size = 1
#                     elif isinstance(x, list):
#                         batch_size = x[0].shape[0]
#                     elif isinstance(x, dict):
#                         batch_size = list(x.values())[0].shape[0]
#                     else:
#                         batch_size = x.shape[0]
#                     batch_logs['batch'] = batch_index
#                     batch_logs['size'] = batch_size
#                     callbacks.on_batch_begin(batch_index, batch_logs)
#                     outs = self.train_on_batch(x, y)

#                     outs = to_list(outs)
#                     for l, o in zip(out_labels, outs):
#                         batch_logs[l] = o.astype(np.float64)

#                     callbacks.on_batch_end(batch_index, batch_logs)

#                     batch_index += 1
#                     steps_done += 1
                    
#                     # Epoch finished.
#                     if steps_done >= steps_per_epoch and do_validation:
#                         print('\nvalidation_start\n')
#                         val_outs = self.evaluate_generator(val_enqueuer_gen,
#                                                            steps=validation_steps, 
#                                                            max_queue_size=1,
#                                                            workers=0,
#                                                            use_multiprocessing=False,
#                                                            verbose=0)
                        
                        
# #                         if val_gen:
# #                         else:
# #                             # No need for try/except because
# #                             # data has already been validated.
# #                             val_outs = model.evaluate(
# #                                 val_x, val_y,
# #                                 batch_size=batch_size,
# #                                 sample_weight=val_sample_weights,
# #                                 verbose=0)
#                         val_outs = to_list(val_outs)
#                         # Same labels assumed.
#                         for l, o in zip(out_labels, val_outs):
#                             epoch_logs['val_' + l] = o
#                         print('\nvalidation_done\n')

#                     if callback_model.stop_training:
#                         break
#                 epoch += 1
                
#                 callbacks.on_epoch_end(epoch, epoch_logs)
#                 val_data.on_epoch_end()
#                 self.on_epoch_end(epoch)
                
#                 if save_frequency is not None:
#                     if epoch // save_frequency * save_frequency == epoch:
#                         self.save(filepath=self.model_save_dir, is_compile=False)
#                         self.save_history(epoch=epoch, verbose=False)
#                 if callback_model.stop_training:
#                     break
#         finally:
#             try:
#                 if enqueuer is not None:
#                     enqueuer.stop()
#             finally:
#                 if val_enqueuer is not None:
#                     val_enqueuer.stop()

#         callbacks.on_train_end()
        
#         if self.best_model_save: self.load(self.model_save_dir)
#         else: self.save(filepath=self.model_save_dir)
#         return self.history
        
#     def evaluate_generator(self, generator,
#                            steps=None,
#                            max_queue_size=10,
#                            workers=0,
#                            use_multiprocessing=False,
#                            verbose=0):
#         """Modified evaluate_generator from keras.training_generator.evaluate_generator"""
#         model = self.wae_model
#         model._make_test_function()

#         if hasattr(model, 'metrics'):
#             for m in model.stateful_metric_functions:
#                 m.reset_states()
#             stateful_metric_indices = [
#                 i for i, name in enumerate(model.metrics_names)
#                 if str(name) in model.stateful_metric_names]
#         else:
#             stateful_metric_indices = []

#         steps_done = 0
#         outs_per_batch = []
#         batch_sizes = []
#         is_sequence = isinstance(generator, Sequence)
#         if not is_sequence and use_multiprocessing and workers > 1:
#             warnings.warn(
#                 UserWarning('Using a generator with `use_multiprocessing=True`'
#                             ' and multiple workers may duplicate your data.'
#                             ' Please consider using the`keras.utils.Sequence'
#                             ' class.'))
#         if steps is None:
#             if is_sequence:
#                 steps = len(generator)
#             else:
#                 raise ValueError('`steps=None` is only valid for a generator'
#                                  ' based on the `keras.utils.Sequence` class.'
#                                  ' Please specify `steps` or use the'
#                                  ' `keras.utils.Sequence` class.')
#         enqueuer = None

#         try:
#             if workers > 0:
#                 if is_sequence:
#                     enqueuer = OrderedEnqueuer(
#                         generator,
#                         use_multiprocessing=use_multiprocessing)
#                 else:
#                     enqueuer = GeneratorEnqueuer(
#                         generator,
#                         use_multiprocessing=use_multiprocessing)
#                 enqueuer.start(workers=workers, max_queue_size=max_queue_size)
#                 output_generator = enqueuer.get()
#             else:
#                 if is_sequence:
#                     output_generator = iter_sequence_infinite(generator)
#                 else:
#                     output_generator = generator

#             if verbose == 1:
#                 progbar = Progbar(target=steps)

#             while steps_done < steps:
#                 generator_output = next(output_generator)      
#                 ## TODO : sample weight
#                 if not hasattr(generator_output, '__len__'):
#                     raise ValueError('Output of generator should be a tuple '
#                                      '(x, y). Found: ' +
#                                      str(generator_output))
#                 if len(generator_output) == 2:
#                     x, y = generator_output
#                 else:
#                     raise ValueError('Output of generator should be a tuple '
#                                      '(x, y). Found: ' +
#                                      str(generator_output))
#                 outs = self.test_on_batch(x, y)
#                 outs = to_list(outs)
#                 outs_per_batch.append(outs)
#                 if x is None or len(x) == 0:
#                     # Handle data tensors support when no input given
#                     # step-size = 1 for data tensors
#                     batch_size = 1
#                 elif isinstance(x, list):
#                     batch_size = x[0].shape[0]
#                 elif isinstance(x, dict):
#                     batch_size = list(x.values())[0].shape[0]
#                 else:
#                     batch_size = x.shape[0]
#                 if batch_size == 0:
#                     raise ValueError('Received an empty batch. '
#                                      'Batches should contain '
#                                      'at least one item.')
#                 steps_done += 1
#                 batch_sizes.append(batch_size)
#                 if verbose == 1:
#                     progbar.update(steps_done)

#         finally:
#             if enqueuer is not None:
#                 enqueuer.stop()
                
#         averages = []
#         for i in range(len(outs)):
#             if i not in stateful_metric_indices:
#                 averages.append(np.average([out[i] for out in outs_per_batch],
#                                            weights=batch_sizes))
#             else:
#                 averages.append(np.float64(outs_per_batch[-1][i]))
#         return unpack_singleton(averages)
    
#     def on_epoch_end(self, epoch):
#         ## Learning rate schedule
#         if self.lr_schedule == "manual":
#             if epoch == 30:
#                 self.decay = self.decay / 2.
#             if epoch == 50:
#                 self.decay = self.decay / 5.
#             if epoch == 100:
#                 self.decay = self.decay / 10.
#         elif self.lr_schedule == "manual_smooth":
#             enum = epoch
#             decay_t = np.exp(np.log(100.) / enum)
#             self.decay = self.decay / decay_t
#         elif self.lr_schedule == "constant":
#             pass
#         else:
#             raise ValueError("No schedule information with %s" % self.lr_schedule)

#         wae_model = self.parallel_wae_model
#         k.set_value(wae_model.optimizer.lr, k.get_value(wae_model.optimizer.lr) * self.decay)
#         print("epoch %s, lr = %s" % (epoch, float(k.get_value(wae_model.optimizer.lr))))
        
# #     # TODO
# #     def predict_on_batch(self, x, y):
# #     def predict_generator   

#####################################################################################################################
# Gaussian-Mixture Conditional WAE Network
#####################################################################################################################
class ConditionalWAE_GAN(WAE_GAN):
    def __init__(self, log, path_info, network_info, n_label, is_profiling=False):
        super(ConditionalWAE_GAN, self).__init__(log, path_info, network_info, n_label, is_profiling=is_profiling)
        self.TB = ConditionalWAETensorBoardWrapper_GAN
        
    def build_model(self, model_yaml_dir=None, verbose=0):
        """
        verbose
            0: Not show any model
            1: Show AE, Discriminator model
            2: Show all models
        """
        # Load Models : encoder, decoder, discriminator
        if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, self.path_info['model_info']['model_architecture'])

        self.encoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder'], verbose=verbose==2)
        self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], verbose=verbose==2)
        self.discriminator_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], verbose=verbose==2)

        # build blocks
        self.image_shape = self.encoder_model.input_shape[1:]
        self.z_dim = self.encoder_model.output_shape[-1]

        real_image = Input(shape=self.image_shape, name='real_image', dtype='float32')
        cls_info = Input(shape=(self.n_label,), name='class_info_input', dtype='float32')
        cls_info_int = Input(shape=(1,), name='class_info_int_input', dtype='int32')

        prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')
        latent_input = Input(shape=(self.z_dim,), name='latent_input', dtype='float32')

        fake_latent = self.encoder_model(real_image)
        recon_image = self.decoder_model([fake_latent, cls_info])
        self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
        
        # GAN model
        p_z = self.discriminator_model([prior_latent])
        q_z = self.discriminator_model([fake_latent])
        output = Concatenate(name='mlp_concat')([p_z, q_z])
        self.gan_model = Model(inputs=[real_image, prior_latent, cls_info], outputs=[output], name='GAN_model')
        
        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        latent_penalty = Lambda(get_qz_trick_loss, name='latent_penalty')(q_z)
        # latent_penalty = Lambda(get_qz_trick_with_weight_loss, name='latent_penalty')([q_z, cls_info])
        self.main_model = Model(inputs=[real_image, cls_info], outputs=[recon_error, latent_penalty], name='WAE_model')

        # Blur information
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model([prior_latent,cls_info])
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent,cls_info], outputs=[gen_sharpness], name='gen_blurr_model')
        if verbose==2:
            self.log.info('Generative sample blurr model')
            self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            
        # cluster information
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([fake_latent, cls_info_int])
        self.cluster_info_model = Model(inputs=[real_image, cls_info_int], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        if verbose==2:
            self.log.info('Cluster information model')
            self.cluster_info_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        self.train_models_lr = {'discriminator':{'lr':float(self.network_info['model_info']['lr_e_adv']),
                                                 'decay':float(self.network_info['model_info']['lr_e_adv_decay'])},
                                'main':{'lr':float(self.network_info['model_info']['lr_e']),
                                        'decay':float(self.network_info['model_info']['lr_e_decay'])}}
        
        # Pretrain model
#         if self.pretrain: self.pretrain_encoder_model = Model(real_image, fake_latent, name='pretrain_encoder_model')
                    
        if verbose:
            self.log.info('Main model')
            self.main_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            self.log.info('Discriminator model')
            self.gan_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
#             if self.pretrain & verbose==2: 
#                 self.log.info('Pretrain model')
#                 self.pretrain_encoder_model.summary(line_length=200, print_fn=self.log.info)
#                 sys.stdout.flush()
    
    def main_sampler(self, x, y):
        cls_info = to_categorical(y, num_classes=self.n_label)
#         return [x, cls_info], [x, np.zeros(x.shape[0], dtype='float32')]
        return [x, cls_info], [np.zeros(x.shape[0], dtype='float32')] * 2
    
    def discriminator_sampler(self, x, y):
        noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
        cls_info = to_categorical(y, num_classes=self.n_label)
        return [x, noise, cls_info], [np.zeros([x.shape[0],2], dtype='float32')]
    
    def load(self, filepath):
        # TODO: fix
        model_path = self.path_info['model_info']['weight']
        
        loss_list = [self.network_info['model_info']['main_loss'],
                     self.network_info['model_info']['penalty_e'],
                     self.network_info['model_info']['discriminator_loss']]
        load_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in loss_list])
        load_dict['SelfAttention2D'] = SelfAttention2D
        load_dict['get_qz_trick_loss'] = get_qz_trick_loss
        load_dict['get_qz_trick_with_weight_loss'] = get_qz_trick_with_weight_loss
        load_dict['mmd_penalty'] = mmd_penalty
        load_dict['get_b'] = get_b
        load_dict['get_b_estimation_var'] = get_b_estimation_var
        load_dict['get_b_penalty_loss'] = get_b_penalty_loss
        load_dict['mean_reconstruction_l2sq_loss_b'] = mean_reconstruction_l2sq_loss_b
        load_dict['mean_reconstruction_l2sq_loss_e'] = mean_reconstruction_l2sq_loss_e
        load_dict['get_class_mean_by_class_index'] = get_class_mean_by_class_index
        
        self.main_model = load_model("%s/%s_%s" % (filepath, "main", model_path), custom_objects=load_dict)
        self.gan_model = load_model("%s/%s_%s" % (filepath, "discriminator", model_path), custom_objects=load_dict)
        
        for layer in self.main_model.layers:
            if 'encoder_e' in layer.name: self.encoder_model = layer
            if 'decoder' in layer.name: self.decoder_model = layer
            if 'discriminator' in layer.name: self.discriminator_model = layer
                        
        real_image = self.main_model.inputs[0] #Input(shape=self.image_shape, name='real_image', dtype='float32')
        cls_info = self.main_model.inputs[1] #Input(shape=(self.n_label,), name='class_info_input', dtype='float32')
        cls_info_int = Input(shape=(1,), name='class_info_int_input', dtype='int32')

        prior_latent = Input(shape=(self.z_dim,), name='prior_latent', dtype='float32')
        latent_input = Input(shape=(self.z_dim,), name='latent_input', dtype='float32')

        fake_latent = self.encoder_model(real_image)
        recon_image = self.decoder_model([fake_latent, cls_info])
        self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        
        # # GAN model
        p_z = self.discriminator_model([prior_latent])
        q_z = self.discriminator_model([fake_latent])
        output = Concatenate(name='mlp_concat')([p_z, q_z])
        self.gan_model = Model(inputs=[real_image, prior_latent, cls_info], outputs=[output], name='GAN_model')
        
        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        latent_penalty = Lambda(get_qz_trick_loss, name='latent_penalty')(q_z)
        # latent_penalty = Lambda(get_qz_trick_with_weight_loss, name='latent_penalty')([q_z, cls_info])
        self.main_model = Model(inputs=[real_image, cls_info], outputs=[recon_error, latent_penalty], name='WAE_model')

        # Blur information
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model([prior_latent,cls_info])
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent,cls_info], outputs=[gen_sharpness], name='gen_blurr_model')
            
        # cluster information
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([fake_latent, cls_info_int])
        self.cluster_info_model = Model(inputs=[real_image, cls_info_int], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')

        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        
        self.model_compile()
        self.log.info('Main model')
        self.main_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        self.log.info('Discriminator model')
        self.gan_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        
    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.train_on_batch(wx, wy, 
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch([wx[0],y])
        if (self.fixed_noise != None).any(): 
#             gen_blurr = self.gen_blurr_model.predict([self.fixed_noise, self.fixed_cls_info], batch_size=y.shape[0])
            gen_blurr = self.gen_blurr_model.predict([self.fixed_noise, self.fixed_cls_info], batch_size=y.shape[0]//(4*self.number_of_gpu))
        else: raise ValueError('No fixed noise')
        dx, dy = self.discriminator_sampler(x, y)
        d_outs = self.parallel_gan_model.train_on_batch(dx, dy,
                                                        sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return main_outs + [cluster_value, ssw, ssb, n_points_mean, n_l] + [d_outs] + [np.min(gen_blurr)]
        
    def test_on_batch(self, x, y, sample_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.test_on_batch(wx, wy,
                                                           sample_weight=sample_weight, reset_metrics = reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch([wx[0],y])
        if (self.fixed_noise != None).any():
            gen_blurr = self.gen_blurr_model.predict([self.fixed_noise, self.fixed_cls_info], batch_size=y.shape[0]//(4*self.number_of_gpu))
        else: raise ValueError('No fixed noise')
        dx, dy = self.discriminator_sampler(x, y)
        d_outs = self.parallel_gan_model.test_on_batch(dx, dy,
                                                       sample_weight=sample_weight, reset_metrics = reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return main_outs + [cluster_value, ssw, ssb, n_points_mean, n_l] + [d_outs] + [np.min(gen_blurr)]

    def on_train_begin(self, x, y=None):
        real_image_blurriness = self.blurr_model.predict_on_batch([x])
        self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))
        self.fixed_noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
        cls_info = to_categorical(np.sort(np.random.choice(np.arange(self.n_label), x.shape[0], replace=True)), num_classes=self.n_label)
        self.fixed_cls_info = cls_info
        self.log.info('-------------------------------------------------')
        self.log.info("Fixed noise class information")
        cls_table = pd.Series(np.argmax(cls_info, axis=1))
        cls_counts = cls_table.value_counts()
        self.log.info('Images per Class')
        self.log.info('\n%s', cls_counts)
        self.log.info('-------------------------------------------------')
        self.log.info('Summary')
        self.log.info('\n%s', cls_counts.describe())
        self.log.info('-------------------------------------------------')

#####################################################################################################################
# RandomInterceptOAE_GAN Network
#####################################################################################################################
class RandomInterceptOAE_GAN(WAE_GAN):
    def __init__(self, log, path_info, network_info, n_label, is_profiling=False):
        super(RandomInterceptOAE_GAN, self).__init__(log, path_info, network_info, n_label, is_profiling=is_profiling)
        
        self.metrics_names = ['main_loss', 'reconstruction', 'penalty_e', 'penalty_b',
                              'b_j_given_x_j_var', 'b_mean', 'b_var', 
                              'cluster_value','ssw', 'ssb', 'n_points', 'n_label',
                              'discriminator_loss',
                              'sharpness']
        self.TB = RandomInterceptOAETensorBoardWrapper_GAN
        
        self.b_sd = float(network_info['model_info']['b_sd'])
        self.lambda_b = float(network_info['model_info']['lambda_b'])
#         self.lambda_b = k.variable(float(network_info['model_info']['lambda_b']), name='lambda_b')
#         self.lambda_b_decay = float(network_info['model_info']['lambda_b_decay'])
        self.lambda_b_var = float(network_info['model_info']['lambda_b_var'])
        
    def build_model(self, model_yaml_dir=None, verbose=0):
        """
        verbose
            0: Not show any model
            1: Show AE, Discriminator model
            2: Show all models
        """
        # Load Models : encoder, decoder, discriminator
        if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, 
                                                                 self.path_info['model_info']['model_architecture'])
        self.encoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder'], 
                                              verbose=verbose==2)
        self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], 
                                              verbose=verbose==2)
        self.discriminator_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], 
                                                    verbose=verbose==2)
        self.save_models = {"encoder":self.encoder_model,
                            "decoder":self.decoder_model,
                            "discriminator":self.discriminator_model
                           }
        # build blocks
        self.image_shape = self.encoder_model.input_shape[1:]
        self.z_dim = self.encoder_model.output_shape[0][-1]

        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_noise = Input(shape=(self.z_dim,), name='prior_noise_input', dtype='float32')
        prior_latent = Input(shape=(self.z_dim,), name='prior_latent_input', dtype='float32')

        b_input = Input(shape=(self.z_dim,), name='estimated_b_input', dtype='float32')
        noise_input = Input(shape=(self.z_dim,), name='noise_input', dtype='float32')
        sample_b, b = Lambda(get_b, name='get_sample_b')([b_input, cls_info])
        latent = Add(name='add_latent')([sample_b, noise_input])
        self.b_encoder_model = Model([b_input, noise_input, cls_info], [latent, b, sample_b], name='encoder_with_b')
        
        estimate_b, fake_noise = self.encoder_model([real_image])
        fake_latent, b, sample_b = self.b_encoder_model([estimate_b, fake_noise, cls_info])

        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # GAN model
        ## $$(\hat{Z}_{ij} - b_i)|b_i \overset{d}{=} (Z_{ij} - b_i)|b_i$$
        p_z = self.discriminator_model(prior_noise)
        q_z = self.discriminator_model(fake_noise)
        output = Concatenate(name='mlp_concat')([p_z, q_z]) ## TODO : fix..
        self.gan_model = Model(inputs=[real_image, cls_info, prior_noise], outputs=[output], name='GAN_model')

        # WAE model
        prior_b = Input(shape=(self.z_dim,), name='prior_b_input', dtype='float32')
        b_penalty = Lambda(get_b_penalty_loss, name='b_penalty',
                           arguments={'sigma':self.b_sd, 'zdim':self.z_dim, 'kernel':'IMQ', 'p_z':'normal'})([prior_b, b])
        b_estimation_var = Lambda(get_b_estimation_var,  name='b_estimation_var')([estimate_b, sample_b, cls_info]) #, q_z])
        
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
#         latent_penalty = Lambda(get_qz_trick_loss, name='latent_penalty')(q_z)
        latent_penalty = Lambda(get_qz_trick_with_weight_loss, name='latent_penalty')([q_z, cls_info])
        self.main_model = Model(inputs=[real_image, cls_info, prior_b], 
                                outputs=[recon_error, latent_penalty, b_penalty, b_estimation_var], name='main_model')

        # Blur information
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        if verbose==2:
            self.log.info('Generative sample blurr model')
            self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # cluster information
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([fake_latent, cls_info])
        self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        if verbose==2:
            self.log.info('Cluster information model')
            self.cluster_info_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
    
        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        self.train_models_lr = {'discriminator':{'lr':float(self.network_info['model_info']['lr_e_adv']),
                                                 'decay':float(self.network_info['model_info']['lr_e_adv_decay'])},
                                'main':{'lr':float(self.network_info['model_info']['lr_e']),
                                        'decay':float(self.network_info['model_info']['lr_e_decay'])}}
        
#         # Pretrain model
#         if self.pretrain: self.pretrain_encoder_model = Model(real_image, fake_latent, name='pretrain_encoder_model')

        if verbose:
            self.log.info('Main model')
            self.main_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            self.log.info('Discriminator model')
            self.gan_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
#             if self.pretrain & verbose==2: 
#                 self.log.info('Pretrain model')
#                 self.pretrain_encoder_model.summary(line_length=200, print_fn=self.log.info)
#                 sys.stdout.flush()
                
    def model_compile(self, verbose=0):
        self.log.info('Start models compile.')
        if self.network_info['model_info']['optimizer'] =='adam':
            optimizer_e = getattr(keras.optimizers,
                                  self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['main']['lr'],
                                                                                beta_1=float(self.network_info['model_info']['lr_e_beta1']))
            optimizer_e_adv = getattr(keras.optimizers,
                                      self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['discriminator']['lr'],
                                                                                    beta_1=float(self.network_info['model_info']['lr_e_adv_beta1']))
        else: 
            optimizer_e = getattr(keras.optimizers,
                                  self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['main']['lr'])
            optimizer_e_adv = getattr(keras.optimizers,
                                      self.network_info['model_info']['optimizer'])(lr=self.train_models_lr['discriminator']['lr'])
            
        # GAN model compile
        self.encoder_model.trainable = False
        self.decoder_model.trainable = False
        self.discriminator_model.trainable = True
        self.parallel_gan_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['discriminator_loss']), 
                                        optimizer=optimizer_e_adv, options=self.run_options, run_metadata=self.run_metadata)
        
        # WAE model compile
        self.encoder_model.trainable = True
        self.decoder_model.trainable = True
        self.discriminator_model.trainable = False
        self.parallel_main_model.compile(loss={'mean_recon_error':getattr(loss_and_metric, self.network_info['model_info']['main_loss']), 
                                               'latent_penalty':getattr(loss_and_metric, self.network_info['model_info']['penalty_e']),
                                               'b_penalty':getattr(loss_and_metric, 'first_penalty_loss'),
                                               'b_estimation_var':getattr(loss_and_metric, 'first_penalty_loss'),
                                              },
                                         loss_weights=[1., self.lambda_e, self.lambda_b, self.lambda_b_var],
                                         optimizer=optimizer_e, options=self.run_options, run_metadata=self.run_metadata)
        
#         if self.pretrain:
#             self.encoder_model.trainable = True
#             self.pretrain_encoder_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['pretrain_loss']),
#                                                 optimizer=optimizer)
        if verbose:
#             for name, model in self.train_models.items():
            for name, model in self.parallel_train_models.items():
                self.log.info('%s model' % name)
                model.summary(line_length=200, print_fn=self.log.info)
                sys.stdout.flush()
            self.log.info('Model compile done.')
        
    def save(self, filepath, is_compile=True, overwrite=True, include_optimizer=True):
        model_path = self.path_info['model_info']['weight']
#         self.lambda_b = k.get_value(self.lambda_b)
#         self.model_compile()
#         for name, model in self.parallel_train_models.items():
        for name, model in self.save_models.items():
            model.save("%s/%s_%s" % (filepath, name, model_path), overwrite=overwrite, include_optimizer=include_optimizer)
        self.log.debug('Save model at %s' % filepath)
#         self.lambda_b = k.variable(self.lambda_b, name='lambda_b')
#         self.model_compile()
        
    def load(self, filepath):
        model_path = self.path_info['model_info']['weight']
        
        loss_list = [self.network_info['model_info']['main_loss'],
                     self.network_info['model_info']['penalty_e'],
                     self.network_info['model_info']['discriminator_loss']]
        load_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in loss_list])
        load_dict['SelfAttention2D'] = SelfAttention2D
        load_dict['get_qz_trick_loss'] = get_qz_trick_loss
        load_dict['get_qz_trick_with_weight_loss'] = get_qz_trick_with_weight_loss
        load_dict['mmd_penalty'] = mmd_penalty
        load_dict['get_b'] = get_b
        load_dict['get_b_estimation_var'] = get_b_estimation_var
        load_dict['get_b_penalty_loss'] = get_b_penalty_loss
        load_dict['mean_reconstruction_l2sq_loss'] = mean_reconstruction_l2sq_loss
        load_dict['get_class_mean_by_class_index'] = get_class_mean_by_class_index
        
        # TODO : fix save & load
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder", model_path), overwrite=False)
        self.encoder_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "decoder", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), overwrite=False)
        self.decoder_model.load_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "decoder", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "discriminator", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), overwrite=False)
        self.discriminator_model.load_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "discriminator", model_path))
                        
        self.image_shape = self.encoder_model.input_shape[1:]
        self.z_dim = self.encoder_model.output_shape[0][-1]
        
        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_noise = Input(shape=(self.z_dim,), name='prior_noise_input', dtype='float32')
        prior_latent = Input(shape=(self.z_dim,), name='prior_latent_input', dtype='float32')
        
        b_input = Input(shape=(self.z_dim,), name='estimated_b_input', dtype='float32')
        noise_input = Input(shape=(self.z_dim,), name='noise_input', dtype='float32')
        sample_b, b = Lambda(get_b, name='get_sample_b')([b_input, cls_info])
        latent = Add(name='add_latent')([sample_b, noise_input])
        self.b_encoder_model = Model([b_input, noise_input, cls_info], [latent, b, sample_b], name='encoder_with_b')
        
        estimate_b, fake_noise = self.encoder_model([real_image])
        fake_latent, b, sample_b = self.b_encoder_model([estimate_b, fake_noise, cls_info])
        
        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        
        p_z = self.discriminator_model(prior_noise)
        q_z = self.discriminator_model(fake_noise)
        output = Concatenate(name='mlp_concat')([p_z, q_z]) ## TODO : fix..
        self.gan_model = Model(inputs=[real_image, cls_info, prior_noise], outputs=[output], name='GAN_model')

        prior_b = Input(shape=(self.z_dim,), name='prior_b_input', dtype='float32')
        b_penalty = Lambda(get_b_penalty_loss, name='b_penalty',
                           arguments={'sigma':self.b_sd, 'zdim':self.z_dim, 'kernel':'IMQ', 'p_z':'normal'})([prior_b, b])
        b_estimation_var = Lambda(get_b_estimation_var,  name='b_estimation_var')([estimate_b, sample_b, cls_info]) #, q_z])
        
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
#         latent_penalty = Lambda(get_qz_trick_loss, name='latent_penalty')(q_z)
        latent_penalty = Lambda(get_qz_trick_with_weight_loss, name='latent_penalty')([q_z, cls_info])
        self.main_model = Model(inputs=[real_image, cls_info, prior_b], 
                                outputs=[recon_error, latent_penalty, b_penalty, b_estimation_var], name='main_model')
        
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')

        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                      name='get_cluster_information_by_class_index')([fake_latent, cls_info])
        self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        
        try:
            self.parallel_main_model = multi_gpu_model(self.main_model, gpus=self.number_of_gpu)
            self.parallel_gan_model = multi_gpu_model(self.gan_model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_main_model = self.main_model
            self.parallel_gan_model = self.gan_model
            self.log.info("Training using single GPU or CPU")

        self.train_models = {'discriminator':self.gan_model, 'main':self.main_model}
        self.parallel_train_models = {'discriminator':self.parallel_gan_model, 'main':self.parallel_main_model}
        self.train_models_lr = {'discriminator':{'lr':float(self.network_info['model_info']['lr_e_adv']),
                                                 'decay':float(self.network_info['model_info']['lr_e_adv_decay'])},
                                'main':{'lr':float(self.network_info['model_info']['lr_e']),
                                        'decay':float(self.network_info['model_info']['lr_e_decay'])}}
        
        
        self.model_compile()
        self.log.info('Loaded WAE model')
        self.main_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        self.log.info('Loaded Discriminator model: GAN')
        self.gan_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        

    def discriminator_sampler(self, x, y):
        noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
#         return [x, y, noise], [np.zeros([x.shape[0],2], dtype='float32')]
        return [x, y[:, np.newaxis], noise], [np.zeros([x.shape[0],2], dtype='float32')]
    
    def main_sampler(self, x, y):
        prior_b = self.noise_sampler(x.shape[0], self.z_dim, scale=self.b_sd)
#         return [x, y, prior_b], [x]+[np.zeros(x.shape[0], dtype='float32')]*3
#         return [x, y[:,np.newaxis], prior_b], [x]+[np.zeros(x.shape[0], dtype='float32')]*3
        return [x, y[:,np.newaxis], prior_b], [np.zeros(x.shape[0], dtype='float32')]*4
        
    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.train_on_batch(wx, wy, 
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch(wx[:2])
        if (self.fixed_noise != None).any(): 
            # fixed noise with predicted b
            estimate_b, fake_noise = self.encoder_model.predict_on_batch(wx[0])
            fake_latent, b, sample_b = self.b_encoder_model.predict_on_batch([estimate_b, fake_noise, wx[1]])
            gen_blurr = self.gen_blurr_model.predict([self.fixed_noise + np.repeat(sample_b, self.fixed_noise.shape[0] // sample_b.shape[0], 
                                                                                  axis=0)], 
                                                     batch_size=wx[0].shape[0]//(self.number_of_gpu))
        else: 
            raise ValueError('No fixed noise')
        dx, dy = self.discriminator_sampler(x, y)
        d_outs = self.parallel_gan_model.train_on_batch(dx, dy, 
                                                        sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return (main_outs +
                [np.mean(b), np.mean(np.var(b, axis=0))] +
                [cluster_value, ssw, ssb, n_points_mean, n_l] +
                [d_outs] + [np.min(gen_blurr)])
    
    def test_on_batch(self, x, y, sample_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        wae_outs = self.parallel_main_model.test_on_batch(wx, wy, 
                                                          sample_weight=sample_weight, reset_metrics = reset_metrics)
        ssw, ssb, n_points_mean, n_l = self.cluster_info_model.predict_on_batch(wx[:2])
        if (self.fixed_noise != None).any(): 
            ## fixed noise with trained b
            estimate_b, fake_noise = self.encoder_model.predict_on_batch(wx[0])
            fake_latent, b, sample_b = self.b_encoder_model.predict_on_batch([estimate_b, fake_noise, wx[1]])
            gen_blurr = self.gen_blurr_model.predict([self.fixed_noise + np.repeat(sample_b, self.fixed_noise.shape[0] // sample_b.shape[0], 
                                                                                  axis=0)], 
                                                     batch_size=wx[0].shape[0]//(self.number_of_gpu))
        else: 
            raise ValueError('No fixed noise')
        dx, dy = self.discriminator_sampler(x, y)
        d_outs = self.parallel_gan_model.test_on_batch(dx, dy, 
                                                       sample_weight=sample_weight, reset_metrics = reset_metrics)
        cluster_value = ssw/(ssb+1e-14)
        return (wae_outs +
                [np.mean(b), np.mean(np.var(b, axis=0))] +
                [cluster_value, ssw, ssb, n_points_mean, n_l] +
                [d_outs] + [np.min(gen_blurr)])
    
    def on_train_begin(self, x):
        real_image_blurriness = self.blurr_model.predict_on_batch(x)
        self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))
        self.fixed_noise = self.noise_sampler(x.shape[0], self.z_dim, self.e_sd)
        
#     def pretrain_fit(self, generator):
#         #### TODO....
#         self.log.info("Pretrain start")
#         nprint = 20
#         nsteps = 500 # TODO, dinamix max step (sample size)
#         hist = []
#         for step in range(nsteps):
#             x, y = generator[step]
#             # dynamic sampler...
#             # traininb prograssbars...
#             cls_info = to_categorical(y, num_classes=self.n_label)
#             noise, sample_b = self.mixed_effect_sampler(x.shape[0], cls_info)
#             raise Error
#             pretrain_outs = self.pretrain_encoder_model.train_on_batch(x, noise)
#             hist.append(pretrain_outs)
#             if step // nprint * nprint == step:
#                 print('%03d/%03d [' % (step, nsteps) + '='*(step//nprint+1)+'>'+'.'*(nsteps//nprint-step//nprint)+'] - loss %.5f' % hist[-1], end='\r')
#         print('%03d/%03d [' % (step+1, nsteps) +'='*(step//nprint+1)+'] - loss %.5f' % hist[-1])
#         self.log.info("Pretrain end")
    
    def on_epoch_end(self, epoch):
        for name in self.train_models_lr.keys():
            if self.train_models_lr[name]['decay'] > 0.:
                self.train_models_lr[name]['lr'] = self._update_lr(epoch, lr=self.train_models_lr[name]['lr'],
                                                                   decay=self.train_models_lr[name]['decay'])
                k.set_value(self.parallel_train_models[name].optimizer.lr, self.train_models_lr[name]['lr'])
                
#         # lambda b
#         if self.lambda_b_decay < 1.:
#             k.set_value(self.lambda_b, k.get_value(self.lambda_b) * self.lambda_b_decay)
#             print("epoch %s, lambda_b = %s" % (epoch, k.get_value(self.lambda_b)))
