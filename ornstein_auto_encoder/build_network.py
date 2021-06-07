import time
import json
import sys
import os
import abc
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
from ._build_base_network import *

#####################################################################################################################
# ProductSpaceOAE_GAN Network with HSIC
#####################################################################################################################
class ProductSpaceOAEHSIC_GAN(WAE_GAN):
    def __init__(self, log, path_info, network_info, n_label, is_profiling=False):
        super(ProductSpaceOAEHSIC_GAN, self).__init__(log, path_info, network_info, n_label, is_profiling=is_profiling)
        
        self.metrics_names = ['main_loss', 'reconstruction', 'penalty_e', 'penalty_b', 'penalty_hsic',
                              'discriminator_loss',
                             ]
        self.TB = ProductSpaceOAETensorBoardWrapper_GAN
        
        self.b_sd = float(network_info['model_info']['b_sd'])
        self.lambda_b = float(network_info['model_info']['lambda_b'])
        self.lambda_hsic = float(network_info['model_info']['lambda_hsic'])
        
        try: self.e_weight = float(network_info['model_info']['e_weight'])
        except: self.e_weight = 1.
        try: self.e_train = not ('false' == network_info['model_info']['e_train'].strip().lower())
        except: self.e_train = True
        try: self.b_train = not ('false' == network_info['model_info']['b_train'].strip().lower())
        except: self.b_train = True
        try: self.feature_b = 'true' == network_info['model_info']['feature_b'].strip().lower()
        except: self.feature_b = False
        try: self.reset_e = 'true' == network_info['model_info']['reset_e'].strip().lower()
        except: self.reset_e = False
            
    def build_model(self, model_yaml_dir=None, verbose=0):
        """
        verbose
            0: Not show any model
            1: Show AE, Discriminator model
            2: Show all models
        """
        # Load Models : encoder, decoder, discriminator
        if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, self.path_info['model_info']['model_architecture'])
        self._encoder_base_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_base'], verbose=verbose==2)
        self._encoder_b_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_b'], verbose=verbose==2)
        self._encoder_e_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_e'], verbose=verbose==2)
        self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], verbose=verbose==2)
        self._discriminator_e_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], verbose=verbose==2)
        self.save_models = {"encoder_base":self._encoder_base_model,
                            "encoder_b":self._encoder_b_model,
                            "encoder_e":self._encoder_e_model,
                            "decoder":self.decoder_model,
                            "discriminator":self._discriminator_e_model
                           }
        
        self._discriminator_e_model.name = 'discriminator_e'

        # build blocks
        self.image_shape = self._encoder_base_model.input_shape[1:]
        if self.feature_b: self.feature_shape = self._encoder_b_model.input_shape[1:]
        self.b_z_dim = self._encoder_b_model.output_shape[-1]
        self.e_z_dim = self._encoder_e_model.output_shape[-1]        
        
        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_e_noise = Input(shape=(self.e_z_dim,), name='prior_e_input', dtype='float32')
        prior_b_noise = Input(shape=(self.b_z_dim,), name='prior_b_input', dtype='float32')
        if self.feature_b: feature_for_b = Input(shape=self.feature_shape, name='feature_b_input', dtype='float32')
        
        # Encoder base
        last_h = self._encoder_base_model([real_image])
        
        # B_i ~ Q_B|X=x^i
        if self.feature_b: b_j_given_x_j = self._encoder_b_model([feature_for_b])
        else: b_j_given_x_j = self._encoder_b_model([last_h])
        sample_b, b_given_x = Lambda(get_b, name='get_b_given_x')([b_j_given_x_j, cls_info])
        if self.feature_b: self.encoder_b_model = Model([feature_for_b, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        else: self.encoder_b_model = Model([real_image, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        
        # E^i_j ~Q_E_0|X_0,B=X^i_j,B_i
        e_given_x_b = self._encoder_e_model([last_h, sample_b])
        if self.feature_b: self.encoder_e_model = Model([real_image, feature_for_b, cls_info], [e_given_x_b], name='encoder_e_model')
        else: self.encoder_e_model = Model([real_image, cls_info], [e_given_x_b], name='encoder_e_model')
        
        # Z^i_j = (B_i, E^i_j)
        b_input = Input(shape=(self.b_z_dim,), name='estimated_b_input', dtype='float32')
        noise_input = Input(shape=(self.e_z_dim,), name='noise_input', dtype='float32')
        if self.e_weight != 1.: noise_weighted = Lambda(lambda x : self.e_weight*x, name='noise_weighted')(noise_input)
        else: noise_weighted = noise_input
        latent = Concatenate(axis=1, name='concat_latent')([b_input, noise_weighted])
        self.z_encoder_model = Model([b_input, noise_input], [latent], name='encoder_z_model')
        
        # Build connections
        if self.feature_b: 
            sample_b, b_given_x, b_j_given_x_j = self.encoder_b_model([feature_for_b, cls_info])
            e_given_x_b = self.encoder_e_model([real_image, feature_for_b, cls_info])
        else: 
            sample_b, b_given_x, b_j_given_x_j = self.encoder_b_model([real_image, cls_info])
            e_given_x_b = self.encoder_e_model([real_image, cls_info])
        fake_latent = self.z_encoder_model([sample_b, e_given_x_b])

        recon_image = self.decoder_model(fake_latent)
        if self.feature_b: self.ae_model = Model(inputs=[real_image, feature_for_b, cls_info], outputs=[recon_image], name='ae_model')
        else: self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # GAN model
        p_e = self._discriminator_e_model(prior_e_noise)
        q_e = self._discriminator_e_model(e_given_x_b)
        output = Concatenate(name='mlp_concat')([p_e, q_e]) ## TODO : fix..
        if self.feature_b: self.gan_model = Model(inputs=[real_image, feature_for_b, cls_info, prior_e_noise], outputs=[output], name='GAN_model')
        else: self.gan_model = Model(inputs=[real_image, cls_info, prior_e_noise], outputs=[output], name='GAN_model')

        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
#         recon_error = Lambda(mean_reconstruction_l2sq_loss_e, name='mean_recon_error')([real_image, recon_image, cls_info])
        penalty_e = Lambda(get_qz_trick_loss, name='penalty_e')(q_e)
        penalty_b = Lambda(get_b_penalty_loss, name='penalty_b',
                           arguments={'sigma':self.b_sd, 'zdim':self.b_z_dim, 'kernel':'RBF', 'p_z':'normal'})([prior_b_noise, b_given_x])      
        penalty_hsic = Lambda(get_hsic, name="penalty_hsic")([e_given_x_b, sample_b])
        if self.feature_b: self.main_model = Model(inputs=[real_image, feature_for_b, cls_info, prior_b_noise], 
                                                   outputs=[recon_error, penalty_e, penalty_b, penalty_hsic], name='main_model')
        else: self.main_model = Model(inputs=[real_image, cls_info, prior_b_noise], 
                                      outputs=[recon_error, penalty_e, penalty_b, penalty_hsic], name='main_model')
            
        # Blur information
        prior_latent = Input(shape=(self.b_z_dim + self.e_z_dim,), name='prior_z_input', dtype='float32')
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
                                              name='get_cluster_information_by_class_index')([b_j_given_x_j, cls_info])
        if self.feature_b: 
            self.cluster_info_model = Model(inputs=[feature_for_b, cls_info], 
                                            outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        else:
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
        
        if verbose:
            self.log.info('Main model')
            self.main_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            self.log.info('Discriminator model')
            self.gan_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
                
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
        
        if self.reset_e:
            self.reset_weights(self._encoder_base_model)
            self.reset_weights(self._encoder_e_model)
            self.reset_weights(self._discriminator_e_model)
            
        # GAN model compile
        self._encoder_b_model.trainable = False
        self._encoder_e_model.trainable = False
        self._encoder_base_model.trainable = False
        self.decoder_model.trainable = False
        self._discriminator_e_model.trainable = self.e_train
        self.parallel_gan_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['discriminator_loss']), 
                                        optimizer=optimizer_e_adv, options=self.run_options, run_metadata=self.run_metadata)
        
        # WAE model compile
        self.decoder_model.trainable = True
        self._encoder_b_model.trainable = self.b_train
        self._encoder_e_model.trainable = self.e_train
        self._encoder_base_model.trainable = self.e_train
        self._discriminator_e_model.trainable = False
        self.parallel_main_model.compile(loss={'mean_recon_error':getattr(loss_and_metric, self.network_info['model_info']['main_loss']), 
                                               'penalty_e':getattr(loss_and_metric, self.network_info['model_info']['penalty_e']),
                                               'penalty_b':getattr(loss_and_metric, self.network_info['model_info']['penalty_b']),
                                               'penalty_hsic':getattr(loss_and_metric, self.network_info['model_info']['penalty_b']),
                                              },
                                         loss_weights=[1., self.lambda_e, self.lambda_b, self.lambda_hsic],
                                         optimizer=optimizer_e, options=self.run_options, run_metadata=self.run_metadata)
            
        if verbose:
            for name, model in self.parallel_train_models.items():
                self.log.info('%s model' % name)
                model.summary(line_length=200, print_fn=self.log.info)
                sys.stdout.flush()
            self.log.info('Model compile done.')
        
    def save(self, filepath, is_compile=True, overwrite=True, include_optimizer=True):
        model_path = self.path_info['model_info']['weight']
        for name, model in self.save_models.items():
            model.save("%s/%s_%s" % (filepath, name, model_path), overwrite=overwrite, include_optimizer=include_optimizer)
        self.log.debug('Save model at %s' % filepath)
        
    def load(self, filepath, verbose=0):
        model_path = self.path_info['model_info']['weight']
        
        loss_list = [self.network_info['model_info']['main_loss'],
                     self.network_info['model_info']['penalty_e'],
                     self.network_info['model_info']['discriminator_loss']]
        load_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in loss_list])
        load_dict['SelfAttention2D'] = SelfAttention2D
        load_dict['get_qz_trick_loss'] = get_qz_trick_loss
        load_dict['get_qz_trick_with_weight_loss'] = get_qz_trick_with_weight_loss
        load_dict['get_hsic'] = get_hsic
        load_dict['mmd_penalty'] = mmd_penalty
        load_dict['get_b'] = get_b
        load_dict['get_b_estimation_var'] = get_b_estimation_var
        load_dict['get_b_penalty_loss'] = get_b_penalty_loss
        load_dict['mean_reconstruction_l2sq_loss'] = mean_reconstruction_l2sq_loss
        load_dict['get_class_mean_by_class_index'] = get_class_mean_by_class_index
        load_dict['concat_with_uniform_sample'] = concat_with_uniform_sample
        load_dict['get_batch_covariance'] = get_batch_covariance
        load_dict['get_mutual_information_from_gaussian_sample'] = get_mutual_information_from_gaussian_sample
        
        # TODO : fix save & load
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_base", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path), overwrite=False)
        self._encoder_base_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_b", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path), overwrite=False)
        self._encoder_b_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_e", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path), overwrite=False)
        self._encoder_e_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "decoder", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), overwrite=False)
        self.decoder_model.load_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "decoder", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "discriminator", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), overwrite=False)
        self._discriminator_e_model.load_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "discriminator", model_path))
            
        self._discriminator_e_model.name = 'discriminator_e'
                        
        # build blocks
        self.image_shape = self._encoder_base_model.input_shape[1:]
        if self.feature_b: self.feature_shape = self._encoder_b_model.input_shape[1:]
        self.b_z_dim = self._encoder_b_model.output_shape[-1]
        self.e_z_dim = self._encoder_e_model.output_shape[-1]        
        
        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_e_noise = Input(shape=(self.e_z_dim,), name='prior_e_input', dtype='float32')
        prior_b_noise = Input(shape=(self.b_z_dim,), name='prior_b_input', dtype='float32')
        if self.feature_b: feature_for_b = Input(shape=self.feature_shape, name='feature_b_input', dtype='float32')
        
        # Encoder base
        last_h = self._encoder_base_model([real_image])
        
        # B_i ~ Q_B|X=x^i
        if self.feature_b: b_j_given_x_j = self._encoder_b_model([feature_for_b])
        else: b_j_given_x_j = self._encoder_b_model([last_h])
        sample_b, b_given_x = Lambda(get_b, name='get_b_given_x')([b_j_given_x_j, cls_info])
        if self.feature_b: self.encoder_b_model = Model([feature_for_b, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        else: self.encoder_b_model = Model([real_image, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        
        # E^i_j ~Q_E_0|X_0,B=X^i_j,B_i
        e_given_x_b = self._encoder_e_model([last_h, sample_b])
        if self.feature_b: self.encoder_e_model = Model([real_image, feature_for_b, cls_info], [e_given_x_b], name='encoder_e_model')
        else: self.encoder_e_model = Model([real_image, cls_info], [e_given_x_b], name='encoder_e_model')
        
        # Z^i_j = (B_i, E^i_j)
        b_input = Input(shape=(self.b_z_dim,), name='estimated_b_input', dtype='float32')
        noise_input = Input(shape=(self.e_z_dim,), name='noise_input', dtype='float32')
        if self.e_weight != 1.: noise_weighted = Lambda(lambda x : self.e_weight*x, name='noise_weighted')(noise_input)
        else: noise_weighted = noise_input
        latent = Concatenate(axis=1, name='concat_latent')([b_input, noise_weighted])
        self.z_encoder_model = Model([b_input, noise_input], [latent], name='encoder_z_model')
        
        # Build connections
        if self.feature_b: 
            sample_b, b_given_x, b_j_given_x_j = self.encoder_b_model([feature_for_b, cls_info])
            e_given_x_b = self.encoder_e_model([real_image, feature_for_b, cls_info])
        else: 
            sample_b, b_given_x, b_j_given_x_j = self.encoder_b_model([real_image, cls_info])
            e_given_x_b = self.encoder_e_model([real_image, cls_info])
        fake_latent = self.z_encoder_model([sample_b, e_given_x_b])

        recon_image = self.decoder_model(fake_latent)
        if self.feature_b: self.ae_model = Model(inputs=[real_image, feature_for_b, cls_info], outputs=[recon_image], name='ae_model')
        else: self.ae_model = Model(inputs=[real_image, cls_info], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # GAN model
        p_e = self._discriminator_e_model(prior_e_noise)
        q_e = self._discriminator_e_model(e_given_x_b)
        output = Concatenate(name='mlp_concat')([p_e, q_e]) ## TODO : fix..
        if self.feature_b: self.gan_model = Model(inputs=[real_image, feature_for_b, cls_info, prior_e_noise], outputs=[output], name='GAN_model')
        else: self.gan_model = Model(inputs=[real_image, cls_info, prior_e_noise], outputs=[output], name='GAN_model')

        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        penalty_e = Lambda(get_qz_trick_loss, name='penalty_e')(q_e)
        penalty_b = Lambda(get_b_penalty_loss, name='penalty_b',
                           arguments={'sigma':self.b_sd, 'zdim':self.b_z_dim, 'kernel':'RBF', 'p_z':'normal'})([prior_b_noise, b_given_x])      
        penalty_hsic = Lambda(get_hsic, name="penalty_hsic")([e_given_x_b, sample_b])
        if self.feature_b: self.main_model = Model(inputs=[real_image, feature_for_b, cls_info, prior_b_noise], 
                                                   outputs=[recon_error, penalty_e, penalty_b, penalty_hsic], name='main_model')
        else: self.main_model = Model(inputs=[real_image, cls_info, prior_b_noise], 
                                      outputs=[recon_error, penalty_e, penalty_b, penalty_hsic], name='main_model')
        
        # Blur information
        prior_latent = Input(shape=(self.b_z_dim + self.e_z_dim,), name='prior_z_input', dtype='float32')
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        
        # cluster information
        ssw, ssb, n_points_mean, n_l = Lambda(self._get_cluster_information_by_class_index, 
                                              name='get_cluster_information_by_class_index')([b_j_given_x_j, cls_info])
        if self.feature_b: 
            self.cluster_info_model = Model(inputs=[feature_for_b, cls_info], 
                                            outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
        else:
            self.cluster_info_model = Model(inputs=[real_image, cls_info], outputs=[ssw, ssb, n_points_mean, n_l], name='get_cluster_info')
            
        self.model_compile()
        self.log.info('Loaded WAE model')
        self.main_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        self.log.info('Loaded Discriminator model: GAN')
        self.gan_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        
    def discriminator_sampler(self, x, y):
        e_noise = self.noise_sampler(y.shape[0], self.e_z_dim, self.e_sd)
        if self.feature_b:
            return [x[0], x[1], y[:, np.newaxis], e_noise], [np.zeros([x[0].shape[0],2], dtype='float32')]
        else:
            return [x, y[:, np.newaxis], e_noise], [np.zeros([x.shape[0],2], dtype='float32')]
        
    def main_sampler(self, x, y):
        b_noise = self.noise_sampler(y.shape[0], self.b_z_dim, self.b_sd) #, dist='spherical_uniform')
        if self.feature_b:
            return [x[0], x[1], y[:,np.newaxis], b_noise], [np.zeros(x[0].shape[0], dtype='float32')]*4
        else:
            return [x, y[:,np.newaxis], b_noise], [np.zeros(x.shape[0], dtype='float32')]*4
        
    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.train_on_batch(wx, wy,
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        dx, dy = self.discriminator_sampler(x, y)
        if self.e_train: d_outs = self.parallel_gan_model.train_on_batch(dx, dy, 
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        else: d_outs = 0
        return (main_outs +
                [d_outs]
               )
    
    def test_on_batch(self, x, y, sample_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.test_on_batch(wx, wy, 
                                                          sample_weight=sample_weight, reset_metrics = reset_metrics)
        dx, dy = self.discriminator_sampler(x, y)
        if self.e_train: d_outs = self.parallel_gan_model.test_on_batch(dx, dy, 
                                                                        sample_weight=sample_weight, reset_metrics = reset_metrics)
        else: d_outs = 0
        return (main_outs +
                [d_outs] 
               )
    
    def get_reference_images(self, generator):
        batches = [generator[i] for i in range(4)]
        self.fixed_classes = np.concatenate([batch[1] for batch in batches])
        if self.feature_b: 
            self.fixed_feature = np.concatenate([batch[0][1] for batch in batches])
            return np.concatenate([batch[0][0] for batch in batches])
        else: 
            return np.concatenate([batch[0] for batch in batches]) 
    
    def on_train_begin(self, x):
        self.fixed_images = x[self.fixed_classes == self.fixed_classes[0]]
        if self.feature_b: self.fixed_feature = self.fixed_feature[self.fixed_classes == self.fixed_classes[0]]
        self.fixed_classes = self.fixed_classes[self.fixed_classes == self.fixed_classes[0]]
        
        real_image_blurriness = self.blurr_model.predict_on_batch(x)
        self.fixed_noise = self.noise_sampler(x.shape[0], self.e_z_dim, self.e_sd)
        self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))

    def on_epoch_end(self, epoch):
        for name in self.train_models_lr.keys():
            if self.train_models_lr[name]['decay'] > 0.:
                self.train_models_lr[name]['lr'] = self._update_lr(epoch, lr=self.train_models_lr[name]['lr'],
                                                                   decay=self.train_models_lr[name]['decay'])
                k.set_value(self.parallel_train_models[name].optimizer.lr, self.train_models_lr[name]['lr'])

#####################################################################################################################
# ProductSpaceOAE using fixed b and HSIC GAN Network
#####################################################################################################################
class ProductSpaceOAEFixedBHSIC_GAN(WAE_GAN):
    def __init__(self, log, path_info, network_info, n_label, is_profiling=False):
        super(ProductSpaceOAEFixedBHSIC_GAN, self).__init__(log, path_info, network_info, n_label, is_profiling=is_profiling)
        
        self.metrics_names = ['main_loss', 'reconstruction', 'penalty_e', 'penalty_hsic',
                              'discriminator_loss',
                             ]
        self.TB = ProductSpaceOAEFixedBTensorBoardWrapper_GAN
        
        self.b_sd = float(network_info['model_info']['b_sd'])
        self.lambda_hsic = float(network_info['model_info']['lambda_hsic'])
        
        try: self.e_weight = float(network_info['model_info']['e_weight'])
        except: self.e_weight = 1.
        try: self.e_train = not ('false' == network_info['model_info']['e_train'].strip().lower())
        except: self.e_train = True
        try: self.reset_e = 'true' == network_info['model_info']['reset_e'].strip().lower()
        except: self.reset_e = False
        
        try: self.feature_b = 'true' == network_info['model_info']['feature_b'].strip().lower()
        except: self.feature_b = False
        
        try: self.fixed_b_path = network_info['training_info']['fixed_b_path'].strip()
        except: raise ValueError("Need to set fixed_b_path")
        
    def build_model(self, model_yaml_dir=None, verbose=0):
        """
        verbose
            0: Not show any model
            1: Show AE, Discriminator model
            2: Show all models
        """
        # Load Models : encoder, decoder, discriminator
        if model_yaml_dir == None: model_yaml_dir = os.path.join(self.model_save_dir, self.path_info['model_info']['model_architecture'])
        self._encoder_base_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_base'], verbose=verbose==2)
        self._encoder_b_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_b'], verbose=verbose==2)
        self._encoder_e_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_encoder_e'], verbose=verbose==2)
        self.decoder_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_decoder'], verbose=verbose==2)
        self._discriminator_e_model = self._load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], verbose=verbose==2)
        self.save_models = {"encoder_base":self._encoder_base_model,
                            "encoder_b":self._encoder_b_model,
                            "encoder_e":self._encoder_e_model,
                            "decoder":self.decoder_model,
                            "discriminator":self._discriminator_e_model
                           }
        
        self._discriminator_e_model.name = 'discriminator_e'        
        
        # build blocks
        self.image_shape = self._encoder_base_model.input_shape[1:]
        if self.feature_b: self.feature_shape = self._encoder_b_model.input_shape[1:]
        self.b_z_dim = self._encoder_b_model.output_shape[-1]
        self.e_z_dim = self._encoder_e_model.output_shape[-1]        
        
        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_e_noise = Input(shape=(self.e_z_dim,), name='prior_e_input', dtype='float32')
        prior_b_noise = Input(shape=(self.b_z_dim,), name='prior_b_input', dtype='float32')
        b_input = Input(shape=(self.b_z_dim,), name='b_input', dtype='float32')
        if self.feature_b: feature_for_b = Input(shape=self.feature_shape, name='feature_b_input', dtype='float32')
        
        # Encoder base
        last_h = self._encoder_base_model([real_image])
        
        # B_i ~ Q_B|X=x^i
        if self.feature_b: b_j_given_x_j = self._encoder_b_model([feature_for_b])
        else: b_j_given_x_j = self._encoder_b_model([last_h])
        sample_b, b_given_x = Lambda(get_b, name='get_b_given_x')([b_j_given_x_j, cls_info])
        if self.feature_b: self.encoder_b_model = Model([feature_for_b, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        else: self.encoder_b_model = Model([real_image, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        
        # E^i_j ~Q_E_0|X_0,B=X^i_j,B_i
        e_given_x_b = self._encoder_e_model([last_h, b_input])
        self.encoder_e_model = Model([real_image, b_input], [e_given_x_b], name='encoder_e_model')
        
        # Z^i_j = (B_i, E^i_j)
        noise_input = Input(shape=(self.e_z_dim,), name='noise_input', dtype='float32')
        if self.e_weight != 1.: noise_weighted = Lambda(lambda x : self.e_weight*x, name='noise_weighted')(noise_input)
        else: noise_weighted = noise_input
        latent = Concatenate(axis=1, name='concat_latent')([b_input, noise_weighted])
        self.z_encoder_model = Model([b_input, noise_input], [latent], name='encoder_z_model')
        
        # Build connections
        e_given_x_b = self.encoder_e_model([real_image, b_input])
        fake_latent = self.z_encoder_model([b_input, e_given_x_b])

        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image, b_input], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # GAN model
        p_e = self._discriminator_e_model(prior_e_noise)
        q_e = self._discriminator_e_model(e_given_x_b)
        output = Concatenate(name='mlp_concat')([p_e, q_e]) ## TODO : fix..
        self.gan_model = Model(inputs=[real_image, b_input, prior_e_noise], outputs=[output], name='GAN_model')

        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        penalty_e = Lambda(get_qz_trick_loss, name='penalty_e')(q_e)
        penalty_hsic = Lambda(get_hsic, name="penalty_hsic")([e_given_x_b, b_input])
        self.main_model = Model(inputs=[real_image, b_input, cls_info], 
                                      outputs=[recon_error, penalty_e, penalty_hsic], name='main_model')
        
        # Blur information
        prior_latent = Input(shape=(self.b_z_dim + self.e_z_dim,), name='prior_z_input', dtype='float32')
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        if verbose==2:
            self.log.info('Generative sample blurr model')
            self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
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
        
        if verbose:
            self.log.info('Main model')
            self.main_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            self.log.info('Discriminator model')
            self.gan_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
                
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
        
        if self.reset_e:
            self.reset_weights(self._encoder_base_model)
            self.reset_weights(self._encoder_e_model)
            self.reset_weights(self._discriminator_e_model)
            
        # GAN model compile
        self._encoder_b_model.trainable = False
        self._encoder_e_model.trainable = False
        self._encoder_base_model.trainable = False
        self.decoder_model.trainable = False
        self._discriminator_e_model.trainable = self.e_train
        self.parallel_gan_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['discriminator_loss']), 
                                        optimizer=optimizer_e_adv, options=self.run_options, run_metadata=self.run_metadata)
        
        # WAE model compile
        self.decoder_model.trainable = True
        self._encoder_b_model.trainable = False
        self._encoder_e_model.trainable = self.e_train
        self._encoder_base_model.trainable = self.e_train
        self._discriminator_e_model.trainable = False
        self.parallel_main_model.compile(loss={'mean_recon_error':getattr(loss_and_metric, self.network_info['model_info']['main_loss']), 
                                               'penalty_e':getattr(loss_and_metric, self.network_info['model_info']['penalty_e']),
                                               'penalty_hsic':getattr(loss_and_metric, self.network_info['model_info']['penalty_b']),
                                              },
                                         loss_weights=[1., self.lambda_e, self.lambda_hsic],
                                         optimizer=optimizer_e, options=self.run_options, run_metadata=self.run_metadata)
        
        if verbose:
            for name, model in self.parallel_train_models.items():
                self.log.info('%s model' % name)
                model.summary(line_length=200, print_fn=self.log.info)
                sys.stdout.flush()
            self.log.info('Model compile done.')
        
    def save(self, filepath, is_compile=True, overwrite=True, include_optimizer=True):
        model_path = self.path_info['model_info']['weight']
        for name, model in self.save_models.items():
            model.save("%s/%s_%s" % (filepath, name, model_path), overwrite=overwrite, include_optimizer=include_optimizer)
        self.log.debug('Save model at %s' % filepath)
        
    def load(self, filepath, verbose=0):
        model_path = self.path_info['model_info']['weight']
        
        loss_list = [self.network_info['model_info']['main_loss'],
                     self.network_info['model_info']['penalty_e'],
                     self.network_info['model_info']['discriminator_loss']]
        load_dict = dict([(loss_name, getattr(loss_and_metric, loss_name)) for loss_name in loss_list])
        load_dict['SelfAttention2D'] = SelfAttention2D
        load_dict['get_qz_trick_loss'] = get_qz_trick_loss
        load_dict['get_qz_trick_with_weight_loss'] = get_qz_trick_with_weight_loss
        load_dict['get_entropy_loss_with_logits'] = get_entropy_loss_with_logits
        load_dict['mmd_penalty'] = mmd_penalty
        load_dict['get_b'] = get_b
        load_dict['get_b_estimation_var'] = get_b_estimation_var
        load_dict['get_b_penalty_loss'] = get_b_penalty_loss
        load_dict['mean_reconstruction_l2sq_loss'] = mean_reconstruction_l2sq_loss
        load_dict['get_class_mean_by_class_index'] = get_class_mean_by_class_index
        load_dict['concat_with_uniform_sample'] = concat_with_uniform_sample
        load_dict['get_batch_covariance'] = get_batch_covariance
        load_dict['get_mutual_information_from_gaussian_sample'] = get_mutual_information_from_gaussian_sample
        
        # TODO : fix save & load
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_base", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path), overwrite=False)
        self._encoder_base_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_base", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_b", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path), overwrite=False)
        self._encoder_b_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_b", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "encoder_e", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path), overwrite=False)
        self._encoder_e_model.load_weights("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "encoder_e", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "decoder", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), overwrite=False)
        self.decoder_model.load_weights("%s/tmp_%s_%s" % (filepath, "decoder", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "decoder", model_path))
        
        tmp_model = load_model("%s/%s_%s" % (filepath, "discriminator", model_path), custom_objects=load_dict)
        tmp_model.save_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), overwrite=False)
        self._discriminator_e_model.load_weights("%s/tmp_%s_%s" % (filepath, "discriminator", model_path), by_name=True)
        os.remove("%s/tmp_%s_%s" % (filepath, "discriminator", model_path))
        
        self._discriminator_e_model.name = 'discriminator_e'
                        
        # build blocks
        self.image_shape = self._encoder_base_model.input_shape[1:]
        if self.feature_b: self.feature_shape = self._encoder_b_model.input_shape[1:]
        self.b_z_dim = self._encoder_b_model.output_shape[-1]
        self.e_z_dim = self._encoder_e_model.output_shape[-1]        
        
        real_image = Input(shape=self.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        prior_e_noise = Input(shape=(self.e_z_dim,), name='prior_e_input', dtype='float32')
        prior_b_noise = Input(shape=(self.b_z_dim,), name='prior_b_input', dtype='float32')
        b_input = Input(shape=(self.b_z_dim,), name='b_input', dtype='float32')
        if self.feature_b: feature_for_b = Input(shape=self.feature_shape, name='feature_b_input', dtype='float32')
        
        # Encoder base
        last_h = self._encoder_base_model([real_image])
        
        # B_i ~ Q_B|X=x^i
        if self.feature_b: b_j_given_x_j = self._encoder_b_model([feature_for_b])
        else: b_j_given_x_j = self._encoder_b_model([last_h])
        sample_b, b_given_x = Lambda(get_b, name='get_b_given_x')([b_j_given_x_j, cls_info])
        if self.feature_b: self.encoder_b_model = Model([feature_for_b, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        else: self.encoder_b_model = Model([real_image, cls_info], [sample_b, b_given_x, b_j_given_x_j], name='encoder_b_model')
        
        # E^i_j ~Q_E_0|X_0,B=X^i_j,B_i
        e_given_x_b = self._encoder_e_model([last_h, b_input])
        self.encoder_e_model = Model([real_image, b_input], [e_given_x_b], name='encoder_e_model')
        
        # Z^i_j = (B_i, E^i_j)
        noise_input = Input(shape=(self.e_z_dim,), name='noise_input', dtype='float32')
        if self.e_weight != 1.: noise_weighted = Lambda(lambda x : self.e_weight*x, name='noise_weighted')(noise_input)
        else: noise_weighted = noise_input
        latent = Concatenate(axis=1, name='concat_latent')([b_input, noise_weighted])
        self.z_encoder_model = Model([b_input, noise_input], [latent], name='encoder_z_model')
        
        # Build connections
        e_given_x_b = self.encoder_e_model([real_image, b_input])
        fake_latent = self.z_encoder_model([b_input, e_given_x_b])

        recon_image = self.decoder_model(fake_latent)
        self.ae_model = Model(inputs=[real_image, b_input], outputs=[recon_image], name='ae_model')
        if verbose==2:
            self.log.info('Auto-Encoder model')
            self.ae_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()

        # GAN model
        p_e = self._discriminator_e_model(prior_e_noise)
        q_e = self._discriminator_e_model(e_given_x_b)
        output = Concatenate(name='mlp_concat')([p_e, q_e]) ## TODO : fix..
        self.gan_model = Model(inputs=[real_image, b_input, prior_e_noise], outputs=[output], name='GAN_model')

        # WAE model
        recon_error = Lambda(mean_reconstruction_l2sq_loss, name='mean_recon_error')([real_image, recon_image])
        penalty_e = Lambda(get_qz_trick_loss, name='penalty_e')(q_e)
        penalty_hsic = Lambda(get_hsic, name="penalty_hsic")([e_given_x_b, b_input])
        self.main_model = Model(inputs=[real_image, b_input, cls_info], 
                                      outputs=[recon_error, penalty_e, penalty_hsic], name='main_model')
        
        # Blur information
        prior_latent = Input(shape=(self.b_z_dim + self.e_z_dim,), name='prior_z_input', dtype='float32')
        self.blurr_model = get_compute_blurriness_model(self.image_shape)
        gen_image = self.decoder_model(prior_latent)
        gen_sharpness = self.blurr_model(gen_image)
        self.gen_blurr_model = Model(inputs=[prior_latent], outputs=[gen_sharpness], name='gen_blurr_model')
        if verbose==2:
            self.log.info('Generative sample blurr model')
            self.gen_blurr_model.summary(line_length=200, print_fn=self.log.info)
            sys.stdout.flush()
            
        self.model_compile()
        self.log.info('Loaded WAE model')
        self.main_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
        self.log.info('Loaded Discriminator model: GAN')
        self.gan_model.summary(line_length=200, print_fn=self.log.info)
        sys.stdout.flush()
    
    def discriminator_sampler(self, x, y):
        e_noise = self.noise_sampler(y.shape[0], self.e_z_dim, self.e_sd)
        ## TODO: not using feature_b
        return [x[0], x[2], e_noise], [np.zeros([y.shape[0],2], dtype='float32')]
        
    def main_sampler(self, x, y):
        ## TODO: not using feature_b
        return [x[0], x[2], y[:,np.newaxis]], [np.zeros(y.shape[0], dtype='float32')]*3
        
    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.train_on_batch(wx, wy,
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)

        dx, dy = self.discriminator_sampler(x, y)
        if self.e_train: d_outs = self.parallel_gan_model.train_on_batch(dx, dy, 
                                                            sample_weight=sample_weight, class_weight=class_weight, reset_metrics=reset_metrics)
        else: d_outs = 0
        return (main_outs + [d_outs]
               )
    
    def test_on_batch(self, x, y, sample_weight=None, reset_metrics=True):
        wx, wy = self.main_sampler(x, y)
        main_outs = self.parallel_main_model.test_on_batch(wx, wy, 
                                                          sample_weight=sample_weight, reset_metrics = reset_metrics)
        dx, dy = self.discriminator_sampler(x, y)
        if self.e_train: d_outs = self.parallel_gan_model.test_on_batch(dx, dy, 
                                                                        sample_weight=sample_weight, reset_metrics = reset_metrics)
        else: d_outs = 0
        return (main_outs + [d_outs]
               )
    
    def get_reference_images(self, generator):
        ## TODO: not using feature_b
        batches = [generator[i] for i in range(4)]
        self.fixed_classes = np.concatenate([batch[1] for batch in batches])
        if self.feature_b: 
            self.fixed_feature = np.concatenate([batch[0][1] for batch in batches])
            return np.concatenate([batch[0][0] for batch in batches])
        else: 
            return np.concatenate([batch for batches in batches])
    
    def on_train_begin(self, x):
        self.fixed_images = x[self.fixed_classes == self.fixed_classes[0]]
        if self.feature_b: self.fixed_feature = self.fixed_feature[self.fixed_classes == self.fixed_classes[0]]
        self.fixed_classes = self.fixed_classes[self.fixed_classes == self.fixed_classes[0]]
        
        real_image_blurriness = self.blurr_model.predict_on_batch(x)
        self.fixed_noise = self.noise_sampler(x.shape[0], self.e_z_dim, self.e_sd)
        self.log.info("Real image's sharpness = %.5f" % np.min(real_image_blurriness))
        
    def on_epoch_end(self, epoch):
        for name in self.train_models_lr.keys():
            if self.train_models_lr[name]['decay'] > 0.:
                self.train_models_lr[name]['lr'] = self._update_lr(epoch, lr=self.train_models_lr[name]['lr'],
                                                                   decay=self.train_models_lr[name]['decay'])
                k.set_value(self.parallel_train_models[name].optimizer.lr, self.train_models_lr[name]['lr'])