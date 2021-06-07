import os
import sys
import json
import time
import numpy as np
import random
import gc
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.python.client import timeline

from . import logging_daily
from . import configuration
from . import loss_and_metric
from . import readers
from . import samplers
from . import build_network

def training(argdict, log, nrepeat=None):
    # max_queue_size=10
    # workers=5
    # use_multiprocessing=False
    # is_profiling = False
    max_queue_size=50
    workers=18
    use_multiprocessing=True
    is_profiling = False
    
    ### Configuration #####################################################################################
    config_data = configuration.Configurator(argdict['path_info'][0], log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(argdict['network_info'][0], log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()

    ### Training hyperparameter ##########################################################################
    model_save_dir = path_info['model_info']['model_dir']
    warm_start= network_info['training_info']['warm_start'] == 'True'
    warm_start_model = network_info['training_info']['warm_start_model']
    try: save_frequency=int(network_info['training_info']['save_frequency'])
    except: save_frequency=None
        
    if nrepeat is not None:
        if nrepeat == 0 and not os.path.exists('%s/*.h5' % (model_save_dir)):
            warm_start = False
            warm_start_model = ''
    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())  
    try: reader_mode = path_info['data_info']['mode'].strip()
    except: reader_mode = 'train'
    reader = reader_class(log, path_info, network_info, mode=reader_mode, verbose=True)

    if warm_start:
        train_tot_idxs_path = os.path.join(warm_start_model, path_info['model_info']['train_tot_idxs'])
        test_tot_idxs_path = os.path.join(warm_start_model, path_info['model_info']['test_tot_idxs'])
        train_idx = np.load(train_tot_idxs_path)
        test_idx = np.load(test_tot_idxs_path)
    else:
        cv_index = reader.get_cv_index(nfold=5)
        train_idx, test_idx = next(cv_index)
        ## check except class
        except_class = np.array(network_info['model_info']['except_class'].split(','))
        if except_class.shape[0] > 0 and except_class[0] != '': train_idx = reader.except_class(train_idx, except_class)
        ## check imbalance
        if network_info['model_info']['minarity_group_size'] == 'None': minarity_group_size = None
        else: minarity_group_size = float(network_info['model_info']['minarity_group_size'])
        if network_info['model_info']['minarity_ratio'] == 'None': minarity_ratio = None
        else: minarity_ratio = float(network_info['model_info']['minarity_ratio'])    
        if minarity_group_size != None and minarity_ratio != None: train_idx = reader.handle_imbalance(train_idx, minarity_group_size, minarity_ratio)
    ## save
    train_tot_idxs_path = os.path.join(model_save_dir, path_info['model_info']['train_tot_idxs'])
    test_tot_idxs_path = os.path.join(model_save_dir, path_info['model_info']['test_tot_idxs'])
    np.save(train_tot_idxs_path, train_idx)
    np.save(test_tot_idxs_path, test_idx)

    ### Sampler ##########################################################################################
    log.info('-----------------------------------------------------------------')
    # Training data sampler
    log.info('Construct training data sampler')
    train_sampler_class = getattr(samplers, network_info['training_info']['sampler_class'].strip())
    train_sampler = train_sampler_class(log, train_idx, reader, network_info['training_info'], verbose=True)
    train_sampler.set_probability_vector()
    train_generator = samplers.train_generator(train_sampler)

    # Validation data sampler
    log.info('Construct validation data sampler')
    validation_sampler_class = getattr(samplers, network_info['validation_info']['sampler_class'].strip())
    validation_sampler = validation_sampler_class(log, test_idx, reader, network_info['validation_info'], verbose=True)
    validation_sampler.set_probability_vector()
    validation_generator = samplers.train_generator(validation_sampler)
    # validation_generator = samplers.evaluation_generator(validation_sampler)

    ### Bulid network ####################################################################################
    log.info('-----------------------------------------------------------------')
    network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
    GAN_network = network_class(log, path_info, network_info, n_label=reader.get_n_label(), is_profiling=is_profiling)
    GAN_network.build_model(verbose=2)
    GAN_network.model_compile(verbose=1)

    ### Training #########################################################################################
    log.info('-----------------------------------------------------------------')
    log.info('Computing start')
    starttime = time.time()

    hist = GAN_network.fit_generator(train_generator,
                                     steps_per_epoch=len(train_generator),
                                     epochs = int(network_info['training_info']['epochs']),
                                     verbose=1, 
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_generator),
                                     max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
                                     shuffle=True, initial_epoch=0,
                                     warm_start=warm_start,
                                     warm_start_model=warm_start_model,
                                     save_frequency=save_frequency)

    sys.stdout.flush()
    log.info('Compute time : {}'.format(time.time()-starttime))

    # Save history
    GAN_network.save_history()

    # Profiling
    if is_profiling:
        tl = timeline.Timeline(GAN_network.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('%s/timeline.json' % GAN_network.model_save_dir, 'w') as f:
            f.write(ctf)
    log.info('Computing End')
    log.info('-----------------------------------------------------------------')