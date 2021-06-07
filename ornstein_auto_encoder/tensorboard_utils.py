import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import keras.backend as k
from keras.callbacks import TensorBoard

from keras.engine.training_utils import is_sequence, iter_sequence_infinite, should_run_validation

#########################################################################################################################
# TensorBoardWrapper for GAN
#########################################################################################################################
class TensorBoardWrapper_GAN(TensorBoard):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1, tb_data_batch_size=None,  **kwargs):
        super(TensorBoardWrapper_GAN, self).__init__(**kwargs)
        self.write_weights_histogram = write_weights_histogram
        self.write_weights_images = write_weights_images
        self.tb_data_steps = tb_data_steps
        self.tb_data_batch_size = tb_data_batch_size 
        if self.tb_data_batch_size != None: self.batch_size = tb_data_batch_size
    
    def tile_patches(self, x, nrows=10, ncols=10):
        shape = k.int_shape(x)
        x = k.reshape(x,[nrows,ncols,k.prod(shape[1:-1]),shape[-1]]) #(10,10,28*28,3)
        x = tf.transpose(x, perm=[2,0,1,3])
        return tf.batch_to_space_nd(x, shape[1:-1], [[0,0],[0,0]])
    
    def set_histogram_summary_model(self, model_name, model, merge_list):
        for layer in model.layers:
            for weight in layer.weights:
                mapped_weight_name = model_name+"_"+ weight.name.replace(':', '_') ###########
                # histogram
                if self.write_weights_histogram: merge_list.append(tf.summary.histogram(mapped_weight_name, weight))
                # gradient histogram
                if self.write_grads:
                    grads = model.optimizer.get_gradients(model.total_loss, weight)

                    def is_indexed_slices(grad):
                        return type(grad).__name__ == 'IndexedSlices'
                    grads = [
                        grad.values if is_indexed_slices(grad) else grad
                        for grad in grads]
                    merge_list.append(tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads))
                #################################################################################
                # weight image
                if self.write_weights_images:
                    w_img = tf.squeeze(weight)
                    shape = k.int_shape(w_img)
                    if len(shape) == 2:  # dense layer kernel case
                        if shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                            shape = k.int_shape(w_img)
                        w_img = tf.reshape(w_img, [1,
                                                   shape[0],
                                                   shape[1],
                                                   1])
                    elif len(shape) == 3:  # 1D convnet case
                        if k.image_data_format() == 'channels_last':
                            # switch to channels_first to display
                            # every kernel as a separate image
                            w_img = tf.transpose(w_img, perm=[2, 0, 1])
                            shape = k.int_shape(w_img)
                        w_img = tf.reshape(w_img, [shape[0],
                                                   shape[1],
                                                   shape[2],
                                                   1])
                    elif len(shape) == 4: # 2D convnet case
                        if k.image_data_format() == 'channels_last':
                            # switch to channels_first to display
                            # every kernel filter as a separate image
                            w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
                            shape = k.int_shape(w_img)
                        w_img = tf.reshape(w_img, [shape[0]*shape[1],
                                                   shape[2],
                                                   shape[3],
                                                   1])
                    elif len(shape) == 5: # conv3D
                        # input_dim * output_dim*depth, width, hieght
                        w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
                        shape = K.int_shape(w_img)
                        w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
                                                   shape[3],
                                                   shape[4],
                                                   1])
                    elif len(shape) == 1:  # bias case
                        w_img = tf.reshape(w_img, [1,
                                                   shape[0],
                                                   1,
                                                   1])
                    else:
                        # not possible to handle 3D convnets etc.
                        continue

                    shape = k.int_shape(w_img)
                    assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                    merge_list.append(tf.summary.image('weight_'+mapped_weight_name, w_img))
                #################################################################################
            if self.write_weights_histogram:
                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            merge_list.append(tf.summary.histogram('{}_{}_out_{}'.format(model_name, layer.name, i), output))
                    else:
                        merge_list.append(tf.summary.histogram('{}_{}_out'.format(model_name, layer.name), layer.output))
        return merge_list
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.outputs[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))

        if "discriminator" in model_name:
            encoder_model = model.get_layer('encoder_e')
            fake_latent = encoder_model.get_output_at(1)
            prior_latent = model.get_input_at(0)[1]
            # histogram
            merge_list.append(tf.summary.histogram('fake_latent_sample', fake_latent))
            merge_list.append(tf.summary.histogram('prior_latent_sample', prior_latent))

            decoder_model = self.train_models['main'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model(prior_latent), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
        model = self.train_models['discriminator']
        encoder_model = model.get_layer('encoder_e')
        fake_latent = encoder_model.get_output_at(1)
        prior_latent = model.get_input_at(0)[1]
        # prior_latent = model.inputs[1]
        self.embedding_models = ['discriminator']
        ########################

        embeddings_vars = {}
        # self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')

        embedding_input = k.concatenate([fake_latent, prior_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='latent_embedding')
        embeddings_vars['latent'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        # self.assign_embeddings.append(batch)
        self.assign_embeddings = batch
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'latent': self.embeddings_metadata}
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]
        projector.visualize_embeddings(self.writer, config)
        
    def set_model(self, model):
        self.total_model = model
        self.train_models = model.train_models
        self.parallel_train_models = model.parallel_train_models
        if k.backend() == 'tensorflow':
            self.sess = k.get_session()
        self.sampler = {}
        for model_name, model in self.train_models.items():
            try: self.sampler[model_name] = getattr(self.total_model, '%s_sampler' % model_name)
            except: raise ValueError("No %s_sampler in GAN network" % model_name)

        #################################################################################
        if self.histogram_freq and self.merged is None:
            self.merged = {}
            for model_name, model in self.train_models.items():
                merge_list = []
                merge_list = self.set_histogram_summary_model(model_name, model, merge_list)
                if self.write_images:
                    merge_list = self.set_image_summary_model(model_name, model, merge_list)
                if len(merge_list) > 0:
                    self.merged[model_name] = tf.summary.merge(merge_list)

        #################################################################################
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        #################################################################################
        if self.embeddings_freq:
            self.set_embedding_model()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
#         if not self.validation_data and self.histogram_freq:
#             raise ValueError("If printing histograms, validation_data must be "
#                              "provided, and cannot be a generator.")
#         if self.validation_data is None and self.embeddings_freq:
#             raise ValueError("To visualize embeddings, embeddings_data must "
#                              "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch == 1 or epoch % self.histogram_freq == 0:
#                 if is_sequence(self.validation_data):
#                     val_data = iter_sequence_infinite(self.validation_data)
#                     validation_steps = len(self.validation_data)
#                 else:
#                     val_data = self.validation_data
                val_data = self.validation_data
                validation_steps = self.tb_data_steps
                
                tensors = {}
                
#                 for model_name, model in self.train_models.items():
                for model_name, model in self.parallel_train_models.items():
                    ##TODO
                    ## Not support sample_weights
                    tensors[model_name] = (model.inputs +
                                           model.targets)
                    if model.uses_learning_phase: tensors[model_name] += [k.learning_phase()]
                    
                for i in range(validation_steps):
#                     _batch_val = next(val_data)
                    _batch_val = val_data[i]
                    y_idxs = np.argsort(_batch_val[1])
                    if self.tb_data_batch_size != None: y_idxs = y_idxs[:self.tb_data_batch_size]
                    
                    if type(_batch_val[0]) == list:
                        _batch_val = [x[y_idxs] for x in _batch_val[0]], _batch_val[1][y_idxs]
                    else:
                        _batch_val = _batch_val[0][y_idxs], _batch_val[1][y_idxs]
                    step = _batch_val[1].shape[0]
                    
                    for model_name in self.merged.keys():
                        model = self.train_models[model_name]
                        val_x, val_y = self.sampler[model_name](_batch_val[0], _batch_val[1])
                        batch_val = val_x + val_y
                        if model.uses_learning_phase: batch_val += [0.]
                        assert len(batch_val) == len(tensors[model_name])
                        feed_dict = dict(zip(tensors[model_name], batch_val))
                        result = self.sess.run([self.merged[model_name]], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, epoch)
                        
# TODO: Fix
#         if self.embeddings_freq and self.validation_data is not None:
#             if epoch == 1 or epoch % self.embeddings_freq == 0:        
#                 embeddings_data = self.validation_data

#                 for i in range(self.tb_data_steps):
#                     e_x, e_y = embeddings_data[i]
                    e_x, e_y = _batch_val
                    y_idxs = np.argsort(e_y)
                    if self.tb_data_batch_size != None: y_idxs = y_idxs[:self.tb_data_batch_size]
                    
                    if type(_batch_val[0]) == list:
                        e_x, e_y = [x[y_idxs] for x in e_x], e_y[y_idxs]
                    else: 
                        e_x, e_y = e_x[y_idxs], e_y[y_idxs]
                    step = e_y.shape[0]
                    
                    with open("%s/%s"%(self.log_dir, self.embeddings_metadata),'w') as f:
                        f.write("Index\tLabel\tClass\n")
                        for index,label in enumerate(e_y):
                            f.write("%d\t%s\t%s\n" % (index,"fake",label)) # fake
                        for index,label in enumerate(e_y):
                            f.write("%d\t%s\t%s\n" % (len(e_y)+index,"true","NA")) # true
                    
                    feed_dict = {}
                    for model_name in self.embedding_models:
                        s_x, _ = self.sampler[model_name](e_x, e_y)
                        model = self.parallel_train_models[model_name]
                        feed_dict.update(zip(model.inputs, s_x))
                    feed_dict.update({self.batch_id: i, self.step: step*2})

                    if model.uses_learning_phase:
                        feed_dict[k.learning_phase()] = False
                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)
#                 if workers > 0:
#                     val_data.join_end_of_epoch()
                
        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

#########################################################################################################################
class ConditionalWAETensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(ConditionalWAETensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                                   write_weights_images = write_weights_images,
                                                                   tb_data_steps = tb_data_steps, **kwargs)
        
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.outputs[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))

        if "discriminator" in model_name:
            encoder_model = model.get_layer('encoder_e')
            fake_latent = encoder_model.get_output_at(1)
            prior_latent = model.get_input_at(0)[1]
            cls_info = model.get_input_at(0)[2]
            # histogram
            merge_list.append(tf.summary.histogram('fake_latent_sample', fake_latent))
            merge_list.append(tf.summary.histogram('prior_latent_sample', prior_latent))

            decoder_model = self.train_models['main'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model([prior_latent, cls_info]), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
            gen_prototype_image = self.tile_patches(decoder_model([tf.zeros_like(prior_latent), cls_info]), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
        return merge_list

#########################################################################################################################
class RandomInterceptOAETensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(RandomInterceptOAETensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                             write_weights_images = write_weights_images,
                                                             tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.output[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
        if "discriminator" in model_name:
            encoder_model = self.train_models['main'].get_layer('encoder_e')
            b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
            estimate_b, fake_noise = encoder_model.get_output_at(1)
            fake_latent, b, sample_b = b_encoder_model.get_output_at(1)

            prior_noise = model.inputs[2]
            exampler_latent = sample_b + prior_noise

            # histogram
            merge_list.append(tf.summary.histogram('fake_latent_sample', fake_latent))
            merge_list.append(tf.summary.histogram('fake_noise_sample', fake_noise))
            merge_list.append(tf.summary.histogram('b', b))
            merge_list.append(tf.summary.histogram('estimate_b', estimate_b))
            merge_list.append(tf.summary.histogram('prior_noise_sample', prior_noise))
            merge_list.append(tf.summary.histogram('exampler_latent_sample', exampler_latent))

            ## TODO : Generated image with learned random effect & generated iamge with sampled random effect
            decoder_model = self.train_models['main'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model(exampler_latent), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
            gen_prototype_image = self.tile_patches(decoder_model([sample_b]), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
            gen_prototype_estimation_image = self.tile_patches(decoder_model([estimate_b]), nrows, ncols)
            shape = k.int_shape(gen_prototype_estimation_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', gen_prototype_estimation_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         encoder_model = self.train_models['main'].get_layer('encoder_e')
#         b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        model = self.train_models['discriminator']
        encoder_model = model.get_layer('encoder_e')
        estimate_b, fake_noise = encoder_model.get_output_at(1)
        b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        fake_latent, b, sample_b = b_encoder_model.get_output_at(1)
        prior_noise = model.inputs[2]
        prior_b_noise = self.train_models['main'].inputs[2]
        prior_latent = prior_b_noise + prior_noise
        exampler_latent = sample_b + prior_noise
        self.embedding_models = ['discriminator', 'main']
        
        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')

        ###########################
        embedding_input = k.concatenate([fake_latent, prior_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_embedding')
        embeddings_vars['z'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([fake_latent, exampler_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_exampler_embedding')
        embeddings_vars['z_exampler'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([estimate_b, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([sample_b, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch    
        ###########################
        embedding_input = k.concatenate([fake_noise, prior_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'z': self.embeddings_metadata,
                               'z_exampler': self.embeddings_metadata,
                               'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata,
                              }
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
        
#########################################################################################################################
class ProductSpaceOAEv0TensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(ProductSpaceOAEv0TensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                             write_weights_images = write_weights_images,
                                                             tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.output[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            

        if "discriminator" in model_name:
            encoder_e_model = model.get_layer('encoder_e_model')
            encoder_b_model = model.get_layer('encoder_b_model')
            encoder_z_model = model.get_layer('encoder_z_model')
            e_given_x_b = encoder_e_model.get_output_at(1)
            sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
            fake_latent = encoder_z_model.get_output_at(1)
            prior_b_noise = model.inputs[-2]
            prior_noise = model.inputs[-1]
            prior_latent = k.concatenate([prior_b_noise, prior_noise], axis=1)

            # histogram
            merge_list.append(tf.summary.histogram('prior_b', prior_b_noise))
            merge_list.append(tf.summary.histogram('fake_latent_sample', fake_latent))
            merge_list.append(tf.summary.histogram('fake_noise_sample', e_given_x_b))
            merge_list.append(tf.summary.histogram('b', sample_b))
            merge_list.append(tf.summary.histogram('estimate_b', b_j_given_x_j))
            merge_list.append(tf.summary.histogram('prior_noise_sample', prior_noise))
            merge_list.append(tf.summary.histogram('prior_latent_sample', prior_latent))

            ## TODO : Generated image with learned random effect & generated iamge with sampled random effect
            decoder_model = self.train_models['main'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model(k.concatenate([sample_b, prior_noise], axis=1)), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
            gen_prototype_image = self.tile_patches(decoder_model([k.concatenate([sample_b, k.zeros_like(prior_noise)], axis=1)]),
                                                    nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
            gen_prototype_estimation_image = self.tile_patches(decoder_model([k.concatenate([b_j_given_x_j, k.zeros_like(prior_noise)], axis=1)]),
                                                               nrows, ncols)
            shape = k.int_shape(gen_prototype_estimation_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', gen_prototype_estimation_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         encoder_model = self.train_models['main'].get_layer('encoder_e')
#         b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        model = self.train_models['discriminator']
        encoder_e_model = model.get_layer('encoder_e_model')
        encoder_b_model = model.get_layer('encoder_b_model')
        encoder_z_model = model.get_layer('encoder_z_model')
        e_given_x_b = encoder_e_model.get_output_at(1)
        sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
        fake_latent = encoder_z_model.get_output_at(1)
        prior_b_noise = model.inputs[-2]
        prior_noise = model.inputs[-1]
        prior_latent = k.concatenate([prior_b_noise, prior_noise], axis=1)
        exampler_latent = k.concatenate([sample_b, prior_noise], axis=1)
        self.embedding_models = ['discriminator', 'main']
        
        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')

        ###########################
        embedding_input = k.concatenate([fake_latent, prior_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_embedding')
        embeddings_vars['z'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch        
        ###########################
        embedding_input = k.concatenate([fake_latent, exampler_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_exampler_embedding')
        embeddings_vars['z_exampler'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([b_j_given_x_j, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([sample_b, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'z': self.embeddings_metadata,
                               'z_exampler' : self.embeddings_metadata,
                               'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata,
                              }
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
        
        
#########################################################################################################################
class ProductSpaceOAEv1TensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(ProductSpaceOAEv1TensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                             write_weights_images = write_weights_images,
                                                             tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.output[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
            prior_b_noise = model.inputs[-1]
            merge_list.append(tf.summary.histogram('prior_b', prior_b_noise))

        if "discriminator" in model_name:
            model = self.train_models['main']
            encoder_e_model = model.get_layer('encoder_e_model')
            encoder_b_model = self.train_models['main'].get_layer('encoder_b_model')
            encoder_z_model = self.train_models['main'].get_layer('encoder_z_model')
            e_given_x_b = encoder_e_model.get_output_at(1)
            sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
            prior_noise = self.train_models['discriminator'].inputs[-1]
            exampler_sample = encoder_z_model([sample_b, prior_noise])
            prototype_sample = encoder_z_model([sample_b, k.zeros_like(prior_noise)])
            prototype_estimation_sample = encoder_z_model([b_j_given_x_j, k.zeros_like(prior_noise)])
            
            # histogram
            merge_list.append(tf.summary.histogram('fake_noise_sample', e_given_x_b))
            merge_list.append(tf.summary.histogram('b', b_given_x))
            merge_list.append(tf.summary.histogram('estimate_b', b_j_given_x_j))
            merge_list.append(tf.summary.histogram('prior_noise_sample', prior_noise))

            ## TODO : Generated image with learned random effect & generated iamge with sampled random effect
            decoder_model = self.train_models['main'].get_layer('decoder')
#             gen_image = self.tile_patches(decoder_model(k.concatenate([sample_b, prior_noise], axis=1)), nrows, ncols)
            gen_image = self.tile_patches(decoder_model(exampler_sample), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
#             gen_prototype_image = self.tile_patches(decoder_model([k.concatenate([sample_b, k.zeros_like(prior_noise)], axis=1)]),
#                                                     nrows, ncols)
            gen_prototype_image = self.tile_patches(decoder_model(prototype_sample), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
#             gen_prototype_estimation_image = self.tile_patches(decoder_model([k.concatenate([b_j_given_x_j, k.zeros_like(prior_noise)], axis=1)]),
#                                                                nrows, ncols)
            gen_prototype_estimation_image = self.tile_patches(decoder_model(prototype_estimation_sample), nrows, ncols)
            shape = k.int_shape(gen_prototype_estimation_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', gen_prototype_estimation_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         encoder_model = self.train_models['main'].get_layer('encoder_e')
#         b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        model = self.train_models['main']
        encoder_e_model = model.get_layer('encoder_e_model')
        encoder_b_model = model.get_layer('encoder_b_model')
        encoder_z_model = model.get_layer('encoder_z_model')
        e_given_x_b = encoder_e_model.get_output_at(1)
        sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
        fake_latent = encoder_z_model.get_output_at(1)
        prior_noise = self.train_models['discriminator'].inputs[-1]
        prior_b_noise = model.inputs[-1]
        exampler_latent = k.concatenate([sample_b, prior_noise], axis=1)
        self.embedding_models = ['discriminator', 'main']
        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')
        ###########################
        embedding_input = k.concatenate([fake_latent, exampler_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_exampler_embedding')
        embeddings_vars['z_exampler'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([b_j_given_x_j, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([sample_b, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'z_exampler' : self.embeddings_metadata,
                               'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata,
                              }
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)


#########################################################################################################################
class ProductSpaceOAETensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(ProductSpaceOAETensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                             write_weights_images = write_weights_images,
                                                             tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.output[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
            prior_b_noise = model.inputs[-1]
            merge_list.append(tf.summary.histogram('prior_b', prior_b_noise))

        if "discriminator" in model_name:
            model = self.train_models['main']
            encoder_e_model = model.get_layer('encoder_e_model')
            encoder_b_model = self.train_models['main'].get_layer('encoder_b_model')
            encoder_z_model = self.train_models['main'].get_layer('encoder_z_model')
            e_given_x_b = encoder_e_model.get_output_at(1)
            sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
            prior_noise = self.train_models['discriminator'].inputs[-1]
            exampler_sample = encoder_z_model([sample_b, prior_noise])
            prototype_sample = encoder_z_model([sample_b, k.zeros_like(prior_noise)])
            prototype_estimation_sample = encoder_z_model([b_j_given_x_j, k.zeros_like(prior_noise)])
            
            # histogram
            merge_list.append(tf.summary.histogram('fake_noise_sample', e_given_x_b))
            merge_list.append(tf.summary.histogram('b', b_given_x))
            merge_list.append(tf.summary.histogram('estimate_b', b_j_given_x_j))
            merge_list.append(tf.summary.histogram('prior_noise_sample', prior_noise))

            ## TODO : Generated image with learned random effect & generated iamge with sampled random effect
            decoder_model = self.train_models['main'].get_layer('decoder')
#             gen_image = self.tile_patches(decoder_model(k.concatenate([sample_b, prior_noise], axis=1)), nrows, ncols)
            gen_image = self.tile_patches(decoder_model(exampler_sample), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
#             gen_prototype_image = self.tile_patches(decoder_model([k.concatenate([sample_b, k.zeros_like(prior_noise)], axis=1)]),
#                                                     nrows, ncols)
            gen_prototype_image = self.tile_patches(decoder_model(prototype_sample), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
#             gen_prototype_estimation_image = self.tile_patches(decoder_model([k.concatenate([b_j_given_x_j, k.zeros_like(prior_noise)], axis=1)]),
#                                                                nrows, ncols)
            gen_prototype_estimation_image = self.tile_patches(decoder_model(prototype_estimation_sample), nrows, ncols)
            shape = k.int_shape(gen_prototype_estimation_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', gen_prototype_estimation_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         encoder_model = self.train_models['main'].get_layer('encoder_e')
#         b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        model = self.train_models['main']
        encoder_e_model = model.get_layer('encoder_e_model')
        encoder_b_model = model.get_layer('encoder_b_model')
        encoder_z_model = model.get_layer('encoder_z_model')
        e_given_x_b = encoder_e_model.get_output_at(1)
        sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
        fake_latent = encoder_z_model.get_output_at(1)
        prior_noise = self.train_models['discriminator'].inputs[-1]
        prior_b_noise = model.inputs[-1]
        exampler_latent = encoder_z_model([sample_b, prior_noise])
        self.embedding_models = ['discriminator', 'main']
        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')
        ###########################
        embedding_input = k.concatenate([fake_latent, exampler_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_exampler_embedding')
        embeddings_vars['z_exampler'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([b_j_given_x_j, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([sample_b, prior_b_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'z_exampler' : self.embeddings_metadata,
                               'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata,
                              }
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
        


#########################################################################################################################
class ProductSpaceOAEFixedBTensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(ProductSpaceOAEFixedBTensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                             write_weights_images = write_weights_images,
                                                             tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" in model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#                         recon_image = self.tile_patches(model.output[0], nrows, ncols)
#             recon_image = self.tile_patches(model.get_output_at(0)[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
        if "discriminator" in model_name:
            model = self.train_models['main']
            encoder_e_model = model.get_layer('encoder_e_model')
            encoder_z_model = self.train_models['main'].get_layer('encoder_z_model')
            e_given_x_b = encoder_e_model.get_output_at(1)
            sample_b = model.inputs[1]
            prior_noise = self.train_models['discriminator'].inputs[-1]
            exampler_sample = encoder_z_model([sample_b, prior_noise])
            prototype_sample = encoder_z_model([sample_b, k.zeros_like(prior_noise)])
            
            # histogram
            merge_list.append(tf.summary.histogram('fake_noise_sample', e_given_x_b))
            merge_list.append(tf.summary.histogram('b', sample_b))
            merge_list.append(tf.summary.histogram('prior_noise_sample', prior_noise))

            ## TODO : Generated image with learned random effect & generated iamge with sampled random effect
            decoder_model = self.train_models['main'].get_layer('decoder')
#             gen_image = self.tile_patches(decoder_model(k.concatenate([sample_b, prior_noise], axis=1)), nrows, ncols)
            gen_image = self.tile_patches(decoder_model(exampler_sample), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
#             gen_prototype_image = self.tile_patches(decoder_model([k.concatenate([sample_b, k.zeros_like(prior_noise)], axis=1)]),
#                                                     nrows, ncols)
            gen_prototype_image = self.tile_patches(decoder_model(prototype_sample), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         encoder_model = self.train_models['main'].get_layer('encoder_e')
#         b_encoder_model = self.train_models['main'].get_layer('encoder_with_b')
        model = self.train_models['main']
        encoder_e_model = model.get_layer('encoder_e_model')
        encoder_z_model = model.get_layer('encoder_z_model')
        e_given_x_b = encoder_e_model.get_output_at(1)
        sample_b = self.train_models['discriminator'].inputs[1]
        fake_latent = encoder_z_model.get_output_at(1)
        prior_noise = self.train_models['discriminator'].inputs[-1]
        exampler_latent = encoder_z_model([sample_b, prior_noise])
        self.embedding_models = ['discriminator', 'main']
        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')
        ###########################
        embedding_input = k.concatenate([fake_latent, exampler_latent], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='z_given_x_exampler_embedding')
        embeddings_vars['z_exampler'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_noise], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'z_exampler' : self.embeddings_metadata,
                               'e': self.embeddings_metadata,
                              }
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
        
#########################################################################################################################        
class OAETensorBoardWrapper_GAN(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(OAETensorBoardWrapper_GAN, self).__init__(write_weights_histogram = write_weights_histogram,
                                                     write_weights_images = write_weights_images,
                                                     tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main_b" == model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

#             recon_image = self.tile_patches(model.output[0], nrows, ncols)
            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
            prior_b = model.inputs[2]
            merge_list.append(tf.summary.histogram('prior_b', prior_b))
            
#         if "discriminator_b" == model_name:
#             discriminator_b_model = model
#             prior_b = discriminator_b_model.inputs[2]
#             merge_list.append(tf.summary.histogram('prior_b', prior_b))
            
        if "discriminator_e" == model_name:
            discriminator_e_model = model            
            prior_e = discriminator_e_model.inputs[2]  
            merge_list.append(tf.summary.histogram('prior_e', prior_e))
            
            encoder_b_model = self.train_models['main_b'].get_layer('encoder_b_model')
            encoder_e_model = self.train_models['main_b'].get_layer('encoder_e_model')
            sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
            e_given_x_b = encoder_e_model.get_output_at(1)
            
            # histogram
            merge_list.append(tf.summary.histogram('b_j_given_x_j', b_j_given_x_j))
            merge_list.append(tf.summary.histogram('b_given_x', b_given_x))
            merge_list.append(tf.summary.histogram('e_given_x_b', e_given_x_b))
            
            decoder_model = self.train_models['main_b'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model([sample_b, prior_e]), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
            gen_prototype_image = self.tile_patches(decoder_model([sample_b, tf.zeros_like(prior_e)]), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
            generated_prototype_estimation = self.tile_patches(decoder_model([b_j_given_x_j, tf.zeros_like(prior_e)]), nrows, ncols)
            shape = k.int_shape(generated_prototype_estimation)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', generated_prototype_estimation, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
#         discriminator_b_model = self.train_models['discriminator_b']
        discriminator_e_model = self.train_models['discriminator_e']
        main_b_model = self.train_models['main_b']
        encoder_b_model = main_b_model.get_layer('encoder_b_model')
        encoder_e_model = discriminator_e_model.get_layer('encoder_e_model')

        prior_b = main_b_model.inputs[2]
        prior_e = discriminator_e_model.inputs[2]

        sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
        e_given_x_b = encoder_e_model.get_output_at(1)
        
#         self.embedding_models = ['discriminator_b', 'discriminator_e']
        self.embedding_models = ['main_b', 'discriminator_e']

        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')
        # self.batch_id = tf.placeholder(tf.int32, name='embedding_batch_id')
        # self.step = tf.placeholder(tf.int32, name='embedding_step')

        ###########################
        embedding_input = k.concatenate([b_j_given_x_j, prior_b], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch        
        ###########################
#         nsize = tf.shape(b_given_x)[0]
#         embedding_input = k.concatenate([b_given_x, prior_b[:nsize]], axis=0)
#         embedding_size = np.prod(embedding_input.shape[1:])
#         embedding_input = tf.reshape(embedding_input,
#                                      (nsize, int(embedding_size)))
#         shape = (self.tb_data_steps*nsize*2, int(embedding_size))
# #         embedding = tf.Variable(tf.zeros(shape), name='b_given_x_embedding')
#         embedding = tf.Variable(tf.fill(shape, value=0.0, name='b_given_x_embedding'))
#         embeddings_vars['b'] = embedding
#         batch = tf.assign(embedding[self.batch_id*nsize*2:(self.batch_id+1)*nsize*2],
#                           embedding_input)
#         self.assign_embeddings.append(batch)
#         # self.assign_embeddings = batch

#         nsize = tf.shape(b_given_x)[0]
        embedding_input = k.concatenate([sample_b, prior_b], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_e], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata}
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
    
    # TODO: fix for validation with OAEBatchClassSampler

#########################################################################################################################
class OAETensorBoardWrapper_GAN_2(TensorBoardWrapper_GAN):
    def __init__(self, write_weights_histogram = True, write_weights_images=False, tb_data_steps=1,  **kwargs):
        super(OAETensorBoardWrapper_GAN_2, self).__init__(write_weights_histogram = write_weights_histogram,
                                                     write_weights_images = write_weights_images,
                                                     tb_data_steps = tb_data_steps, **kwargs)
    
    def set_image_summary_model(self, model_name, model, merge_list):
        nrows = np.ceil(np.sqrt(self.batch_size)).astype(np.int32)
        ncols = self.batch_size // nrows

        if "main" == model_name:
            input_image = self.tile_patches(model.inputs[0], nrows, ncols)
            shape = k.int_shape(input_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('input_image', input_image, max_outputs=1))

            recon_image = self.tile_patches(model.get_layer('decoder').get_output_at(1), nrows, ncols)
            shape = k.int_shape(recon_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('recon_image', recon_image, max_outputs=1))
            
            prior_b = model.inputs[2]
            merge_list.append(tf.summary.histogram('prior_b', prior_b))
            
        if "discriminator_e" == model_name:
            discriminator_e_model = model            
            prior_e = discriminator_e_model.inputs[2]  
            merge_list.append(tf.summary.histogram('prior_e', prior_e))
            
            encoder_b_model = self.train_models['main'].get_layer('encoder_b_model')
            encoder_e_model = self.train_models['main'].get_layer('encoder_e_model')
            sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
            e_given_x_b = encoder_e_model.get_output_at(1)
            
            # histogram
            merge_list.append(tf.summary.histogram('b_j_given_x_j', b_j_given_x_j))
            merge_list.append(tf.summary.histogram('b_given_x', b_given_x))
            merge_list.append(tf.summary.histogram('e_given_x_b', e_given_x_b))
            
            decoder_model = self.train_models['main'].get_layer('decoder')
            gen_image = self.tile_patches(decoder_model([sample_b, prior_e]), nrows, ncols)
            shape = k.int_shape(gen_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_image', gen_image, max_outputs=1))
            
            gen_prototype_image = self.tile_patches(decoder_model([sample_b, tf.zeros_like(prior_e)]), nrows, ncols)
            shape = k.int_shape(gen_prototype_image)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_image', gen_prototype_image, max_outputs=1))
            
            generated_prototype_estimation = self.tile_patches(decoder_model([b_j_given_x_j, tf.zeros_like(prior_e)]), nrows, ncols)
            shape = k.int_shape(generated_prototype_estimation)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            merge_list.append(tf.summary.image('generated_prototype_estimation_image', generated_prototype_estimation, max_outputs=1))
        return merge_list
    
    def set_embedding_model(self):
        #######################
        # embedding layer
        discriminator_e_model = self.train_models['discriminator_e']
        main_model = self.train_models['main']
        encoder_b_model = main_model.get_layer('encoder_b_model')
        encoder_e_model = discriminator_e_model.get_layer('encoder_e_model')

        prior_b = main_model.inputs[2]
        prior_e = discriminator_e_model.inputs[2]

        sample_b, b_given_x, b_j_given_x_j = encoder_b_model.get_output_at(1)
        e_given_x_b = encoder_e_model.get_output_at(1)
        
        self.embedding_models = ['main', 'discriminator_e']

        ########################
        embeddings_vars = {}
        self.assign_embeddings = []

        self.batch_id = k.variable(0, dtype="int32", name='embedding_batch_id')
        self.step = k.variable(self.batch_size*2, dtype="int32", name='embedding_step')

        ###########################
        embedding_input = k.concatenate([b_j_given_x_j, prior_b], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='b_j_given_x_j_embedding')
        embeddings_vars['b_j'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch        
        ###########################
        embedding_input = k.concatenate([sample_b, prior_b], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='sample_b_embedding')
        embeddings_vars['sample_b'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        embedding_input = k.concatenate([e_given_x_b, prior_e], axis=0)
        embedding_size = np.prod(embedding_input.shape[1:])
        embedding_input = tf.reshape(embedding_input,
                                     (self.step, int(embedding_size)))
        shape = (self.tb_data_steps*self.batch_size*2, int(embedding_size))
        embedding = tf.Variable(tf.zeros(shape), name='e_embedding')
        embeddings_vars['e'] = embedding
        batch = tf.assign(embedding[self.batch_id*self.step:(self.batch_id+1)*self.step],
                          embedding_input)
        self.assign_embeddings.append(batch)
        # self.assign_embeddings = batch
        ###########################
        self.saver = tf.train.Saver(list(embeddings_vars.values()))

        #embeddings_metadata...{}
        embeddings_metadata = {'b_j': self.embeddings_metadata,
                               'sample_b': self.embeddings_metadata,
                               'e': self.embeddings_metadata}
        config = projector.ProjectorConfig()

        for name, tensor in embeddings_vars.items():
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name

            if name in embeddings_metadata:
                embedding.metadata_path = embeddings_metadata[name]

        projector.visualize_embeddings(self.writer, config)
    
    # TODO: fix for validation with OAEBatchClassSampler