import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import random
import tensorflow as tf
# import PIL

seed_value = 123
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

from keras.utils import to_categorical
import keras.backend as k

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
k.set_session(tf.Session(config=config))

sys.path.append('/'.join(os.getcwd().split('/')))
from ornstein_auto_encoder import logging_daily
from ornstein_auto_encoder import configuration
from ornstein_auto_encoder import readers
from ornstein_auto_encoder import samplers
from ornstein_auto_encoder import build_network
from ornstein_auto_encoder.utils import argv_parse
if '1.15' in tf.__version__:
    from ornstein_auto_encoder.fid_v1_15 import get_fid as _get_fid
else:
    from ornstein_auto_encoder.fid import get_fid as _get_fid
from ornstein_auto_encoder.inception_score import get_inception_score as _get_inception_score
#####################################################################################################

def get_fid(images1, images2):
    imgs1 = np.clip(255*((images1).transpose([0,3,1,2]) * 0.5 + 0.5),0,255) #.astype(np.uint8)
    imgs2 = np.clip(255*((images2).transpose([0,3,1,2]) * 0.5 + 0.5),0,255) #.astype(np.uint8)
    return _get_fid(imgs1, imgs2)

def get_is(images, size=100):
    imgs = np.clip(255*(images.transpose([0,3,1,2]) * 0.5 + 0.5),0,255) #.astype(np.uint8)
    return _get_inception_score(imgs, splits=1)[0]

if __name__=='__main__':
    argdict = argv_parse(sys.argv)
    logger = logging_daily.logging_daily(argdict['log_info'][0])
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    log.info('-----------------------------------------------------------------------------------')
    log.info('Evaluate the performance measures for VGGFace2')
    log.info('-----------------------------------------------------------------------------------')
    
    model_path = argdict['model_path'][0].strip()
    try:
        model_aka = argdict['model_aka'][0].strip()
    except:
        model_aka = model_path.split('/')[-1]
    feature_b = True

    path_info_config = argdict['path_info'][0]
    network_info_config = argdict['network_info'][0]
    ##############################################################################################

    # Set hyper-parameter for testing
    config_data = configuration.Configurator(path_info_config, log, verbose=False)
    config_data.set_config_map(config_data.get_section_map())
    config_network = configuration.Configurator(network_info_config, log, verbose=False)
    config_network.set_config_map(config_network.get_section_map())
    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    path_info['model_info']['model_dir'] = model_path
    if network_info['model_info']['network_class'] == 'ProductSpaceOAEFixedBHSIC_GAN': 
        network_info['model_info']['network_class'] == 'ProductSpaceOAEHSIC_GAN'
    if float(network_info['model_info']['e_weight']) == 0.: network_info['model_info']['e_weight'] = '1.'
    if network_info['training_info']['warm_start'] == 'True': 
        network_info['training_info']['warm_start'] = 'False'
        network_info['training_info']['warm_start_model'] = ''
    if network_info['model_info']['augment'] == 'True':
        network_info['model_info']['augment'] = 'False'

    ##############################################################################################
    # Reader
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())  
    reader = reader_class(log, path_info, network_info, mode='train', verbose=True)

    def get_numerics(model_path, model_aka, 
                     path_info_config = "configurations/vggface2/psoae_path_info.cfg",
                     network_info_config = "configurations/vggface2/psoae_network_total_info.cfg",
                     unknown=False, feature_b=False):
        # Set hyper-parameter for testing
        config_data = configuration.Configurator(path_info_config, log, verbose=False)
        config_data.set_config_map(config_data.get_section_map())
        config_network = configuration.Configurator(network_info_config, log, verbose=False)
        config_network.set_config_map(config_network.get_section_map())
        path_info = config_data.get_config_map()
        network_info = config_network.get_config_map()
        path_info['model_info']['model_dir'] = model_path
        if network_info['model_info']['network_class'] == 'ProductSpaceOAEFixedBHSIC_GAN': 
            network_info['model_info']['network_class'] == 'ProductSpaceOAEHSIC_GAN'
        if float(network_info['model_info']['e_weight']) == 0.: network_info['model_info']['e_weight'] = '1.'
        if network_info['training_info']['warm_start'] == 'True': 
            network_info['training_info']['warm_start'] = 'False'
            network_info['training_info']['warm_start_model'] = ''
        if network_info['model_info']['augment'] == 'True':
            network_info['model_info']['augment'] = 'False'


        log.info('-----------------------------------------------------------------')
        unknown = unknown
        log.info('%s: unknown=%s' % (model_aka, unknown))
        log.info('-----------------------------------------------------------------')
        config_data = configuration.Configurator(argdict['path_info'][0], log, verbose=False)
        config_data.set_config_map(config_data.get_section_map())
        config_network = configuration.Configurator(argdict['network_info'][0], log, verbose=False)
        config_network.set_config_map(config_network.get_section_map())

        path_info = config_data.get_config_map()
        network_info = config_network.get_config_map()

        # Set hyper-parameter for testing
        path_info['model_info']['model_dir'] = model_path
        if network_info['model_info']['network_class'] == 'ProductSpaceOAEFixedBHSIC_GAN': 
            network_info['model_info']['network_class'] == 'ProductSpaceOAEHSIC_GAN'
        if float(network_info['model_info']['e_weight']) == 0.: network_info['model_info']['e_weight'] = '1.'
        if network_info['training_info']['warm_start'] == 'True': 
            network_info['training_info']['warm_start'] = 'False'
            network_info['training_info']['warm_start_model'] = ''
        if network_info['model_info']['augment'] == 'True':
            network_info['model_info']['augment'] = 'False'

        ### Bulid network ####################################################################################
        log.info('-----------------------------------------------------------------')
        network_class = getattr(build_network, ''.join(network_info['model_info']['network_class'].strip().split('FixedB')))  
        network = network_class(log, path_info, network_info, n_label=reader.get_n_label())
        network.build_model('./%s/%s' % (model_path,  path_info['model_info']['model_architecture']), verbose=0)
        network.load(model_path)
        log.info('-----------------------------------------------------------------')

        # Training
        test_tot_idxs_path = os.path.join(model_path, path_info['model_info']['test_tot_idxs'])
        test_idx = np.load(test_tot_idxs_path)

        if unknown:
            # Real Test data sampler (not-trained subject)
            new_network_info = copy.deepcopy(network_info)
            new_path_info = copy.deepcopy(path_info)
            new_reader = reader_class(log, new_path_info, new_network_info, mode='test', verbose=False)
            real_test_idx = np.arange(new_reader.get_label().shape[0])
            test_idx = real_test_idx

        log.info('Construct test data sampler')
        validation_sampler_class = getattr(samplers, network_info['validation_info']['sampler_class'].strip())

        if unknown:
            test_sampler = validation_sampler_class(log, test_idx, new_reader, network_info['validation_info'], verbose=False)
        else:
            test_sampler = validation_sampler_class(log, test_idx, reader, network_info['validation_info'], verbose=False)

        tot_sharpness_original = []
        tot_is_original = []

        # tot_reconstruction = []
        tot_gen_fid = []
        tot_gen_is = []
        tot_sharpness_gen = []

        tot_one_shot_gen_fid = []
        tot_one_shot_gen_is = []
        tot_one_shot_sharpness_gen = []

        for nrepeat in range(10):
            log.info('-%d------------------------------------------------' % nrepeat)
            nunit = 30
            nobservations = 300

            picked_y_class = np.random.choice(test_sampler.y_class, nunit, replace=False)
            test_idxs = []
            picked_one_shot_idxs = []
            for yc in picked_y_class:
                try: chosen_observations = np.random.choice(test_sampler.train_idx[test_sampler.y_index.get_loc(yc)], nobservations)
                except: chosen_observations = np.random.choice(test_sampler.train_idx[test_sampler.y_index.get_loc(yc)], nobservations, replace=True)
                test_idxs.append(chosen_observations)
                picked_one_shot_idxs.append(np.random.choice(np.arange(nobservations), 1)[0])
            test_idxs = np.array(test_idxs).flatten()
            picked_one_shot_idxs = np.array(picked_one_shot_idxs)

            x, y = test_sampler.reader.get_batch(test_idxs)

            y_table = pd.Series(y)
            y_index = pd.Index(y)
            y_class = y_table.unique()
            y_table = pd.Series(y)
            y_counts = y_table.value_counts()
            log.info('-------------------------------------------------')
            log.info('Images per Class')
            log.info('\n%s', y_counts)
            log.info('-------------------------------------------------')
            log.info('Summary')
            log.info('\n%s', y_counts.describe())
            log.info('-------------------------------------------------')

            repeated = 300
            esp = 1.
            gen_y_class = np.repeat(y_class, repeated, axis=0)

            try:
                if len(x.shape) == 4: real_img = x
            except: real_img = x[0]

            if 'randomintercept' in network_info['model_info']['network_class'].lower():
                b_sd = float(network_info['model_info']['b_sd'])
                estimate_b, fake_noise = network.encoder_model.predict(x, batch_size=100)
                new_b = np.random.multivariate_normal(np.zeros(network.get_z_dim()), 
                                                      b_sd**2.*np.identity(network.get_z_dim()), 
                                                      y_class.shape[0]).astype(np.float32)
                b = np.array([np.mean(estimate_b[np.random.choice(np.where(y==cls)[0],5)], axis=0) for cls in y_class])
                picked_one_shot_idxs_per_class = np.array([np.where(y==cls)[0][picked_one_shot_idxs[i]] for i, cls in enumerate(y_class)])
                one_shot_b = np.array([estimate_b[picked_one_shot_idxs_per_class[i]] for i, cls in enumerate(y_class)])
                fake_latent = estimate_b + fake_noise
            elif 'productspace' in network_info['model_info']['network_class'].lower():
                wx, wy = network.main_sampler(x,y)

                if feature_b: img, feature, clss, b_noise = wx
                else: img, clss, b_noise = wx
                b_sd = float(network_info['model_info']['b_sd'])

                if feature_b:
                    sample_b, b_given_x, estimate_b  = network.encoder_b_model.predict_on_batch([feature, clss])
                else:
                    sample_b, b_given_x, estimate_b  = network.encoder_b_model.predict_on_batch([img, clss])
                b = np.array([np.mean(estimate_b[np.random.choice(np.where(y==cls)[0],5)], axis=0) for cls in y_class])
                fake_noise = network.encoder_e_model.predict(wx[:-1], batch_size=100)
                new_b = np.random.multivariate_normal(np.zeros(b.shape[-1]), 
                                                      b_sd**2.*np.identity(b.shape[-1]), 
                                                      y_class.shape[0]).astype(np.float32)
                picked_one_shot_idxs_per_class = np.array([np.where(y==cls)[0][picked_one_shot_idxs[i]] for i, cls in enumerate(y_class)])
                one_shot_b = np.array([estimate_b[picked_one_shot_idxs_per_class[i]] for i, cls in enumerate(y_class)])
            else:
                fake_latent = network.encoder_model.predict(real_img, batch_size=100)
                fake_noise = fake_latent

            mean = np.zeros(fake_noise.shape[-1])
            cov = float(network_info['model_info']['e_sd'])**2.*np.identity(fake_noise.shape[-1])
            noise = np.random.multivariate_normal(mean,cov,y_class.shape[0]*repeated).astype(np.float32)

            if 'randomintercept' in network_info['model_info']['network_class'].lower():
                generated_images = network.decoder_model.predict(noise + np.repeat(b, repeated, axis=0),  batch_size=100)
                one_shot_generated_images = network.decoder_model.predict(noise + np.repeat(one_shot_b, repeated, axis=0), batch_size=100)
            elif 'productspace' in network_info['model_info']['network_class'].lower():
                generated_images = network.decoder_model.predict(np.concatenate([np.repeat(b, repeated, axis=0), noise], axis=-1),
                                                                 batch_size=100)
                one_shot_generated_images = network.decoder_model.predict(np.concatenate([np.repeat(one_shot_b, repeated, axis=0), noise], axis=-1),
                                                                          batch_size=100)
            elif 'conditional' in network_info['model_info']['network_class'].lower():
                generated_images = network.decoder_model.predict([noise, to_categorical(gen_y_class, reader.get_n_label())], batch_size=100)
            else:
                generated_images = network.decoder_model.predict(noise, batch_size=100)

            numeric_dict = {}
            origin_sharpness = np.min(network.blurr_model.predict(real_img,batch_size=100))
            gen_sharpness = np.min(network.blurr_model.predict(generated_images,batch_size=100))
            numeric_dict['sharpness_original'] = origin_sharpness
            numeric_dict['original_is'] = get_is(real_img)
            numeric_dict['sharpness_gen'] = gen_sharpness
            numeric_dict['gen_fid'] = get_fid(real_img, generated_images)
            numeric_dict['gen_is'] = get_is(generated_images)

            if 'oae' in network_info['model_info']['network_class'].lower():
                one_shot_gen_sharpness = np.min(network.blurr_model.predict(one_shot_generated_images, batch_size=100))
                numeric_dict['sharpness_one_shot_gen'] = one_shot_gen_sharpness
                numeric_dict['one_shot_gen_fid'] = get_fid(real_img, one_shot_generated_images)
                numeric_dict['one_shot_gen_is'] = get_is(one_shot_generated_images)

            log.info(numeric_dict)
            tot_sharpness_original.append(numeric_dict['sharpness_original'])

            tot_gen_fid.append(numeric_dict['gen_fid'])
            tot_gen_is.append(numeric_dict['gen_is'])
            tot_is_original.append(numeric_dict['original_is'])
            tot_sharpness_gen.append(numeric_dict['sharpness_gen'])

            if 'oae' in network_info['model_info']['network_class'].lower():
                tot_one_shot_gen_fid.append(numeric_dict['one_shot_gen_fid'])
                tot_one_shot_gen_is.append(numeric_dict['one_shot_gen_is'])
                tot_one_shot_sharpness_gen.append(numeric_dict['sharpness_one_shot_gen'])


        log.info('-----------------------------------------------------------------')
        log.info('Results of %s: unknown=%s' % (model_aka, unknown))
        log.info('Original IS: %.3f (\pm %.3f)' % (np.mean(tot_is_original), np.std(tot_is_original)))
        log.info('Original_sharpness: %.3f (\pm %.3f)' % (np.mean(tot_sharpness_original), np.std(tot_sharpness_original)))

        log.info('FID: %.3f (\pm %.3f)' % (np.mean(tot_gen_fid), np.std(tot_gen_fid)))
        log.info('IS: %.3f (\pm %.3f)' % (np.mean(tot_gen_is), np.std(tot_gen_is)))
        log.info('Sharpness: %.3f (\pm %.3f)' % (np.mean(tot_sharpness_gen), np.std(tot_sharpness_gen)))

        if 'oae' in network_info['model_info']['network_class'].lower():
            log.info('One-shot FID: %.3f (\pm %.3f)' % (np.mean(tot_one_shot_gen_fid), np.std(tot_one_shot_gen_fid)))
            log.info('One-shot IS: %.3f (\pm %.3f)' % (np.mean(tot_one_shot_gen_is), np.std(tot_one_shot_gen_is)))
            log.info('One-shot Sharpness: %.3f (\pm %.3f)' % (np.mean(tot_one_shot_sharpness_gen), np.std(tot_one_shot_sharpness_gen)))
    #         tot_numerics = [
    #             tot_sharpness_original,
    #             tot_is_original,
    #             tot_gen_fid,
    #             tot_gen_is,
    #             tot_sharpness_gen,
    #             tot_one_shot_gen_fid,
    #             tot_one_shot_gen_is,
    #             tot_one_shot_sharpness_gen
    #         ]
    #     else:
    #         tot_numerics = [
    #             tot_sharpness_original,
    #             tot_is_original,
    #             tot_gen_fid,
    #             tot_gen_is,
    #             tot_sharpness_gen
    #         ]

    #     if unknown: np.save('./analysis/numerics_%s_unknown.npy' % model_aka, np.array(tot_numerics))
    #     else: np.save('./analysis/numerics_%s.npy' % model_aka, np.array(tot_numerics))
        log.info('-----------------------------------------------------------------------------------')

    ##############################################################################################
    log.info('-----------------------------------------------------------------------------------')
    get_numerics(model_path, model_aka, path_info_config, network_info_config, unknown=False, feature_b=feature_b)
    get_numerics(model_path, model_aka, path_info_config, network_info_config, unknown=True, feature_b=feature_b)
    log.info('-----------------------------------------------------------------------------------')
    log.info('Finished')
    log.info('-----------------------------------------------------------------------------------')