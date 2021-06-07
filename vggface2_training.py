import os, sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

sys.path.append('/'.join(os.path.abspath(__file__).split('/')))
from ornstein_auto_encoder import logging_daily
from ornstein_auto_encoder import configuration
from ornstein_auto_encoder.utils import argv_parse
from ornstein_auto_encoder.training import training
from ornstein_auto_encoder.extract_identity_feature import extract_identity_feature

if __name__ == '__main__':
    argdict = argv_parse(sys.argv)
    
    logger = logging_daily.logging_daily(argdict['log_info'][0])
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    log.info('Argument input')
    for argname, arg in argdict.items():
        log.info('    {}:{}'.format(argname,arg))

    argdict_identity = {'path_info':argdict['path_info'], 'network_info':argdict['network_identity_info'], 'batch_size':['100']}
    argdict_within_unit = {'path_info':argdict['path_info'], 'network_info':argdict['network_within_unit_info']}
    argdict_total = {'path_info':argdict['path_info'], 'network_info':argdict['network_total_info']}
    
    # Alternating optimization
    for nrepeat in range(3):
        log.info('----------------------------------------------------------------------------------------')
        log.info(' %d th alternating ' % nrepeat)
        log.info('----------------------------------------------------------------------------------------')
        log.info('        %dth ideneity encoder update' % nrepeat)
        log.info('----------------------------------------------------------------------------------------')
        # 1) Update the identity encoder
        training(argdict_identity, log, nrepeat=nrepeat)
        extract_identity_feature(argdict_identity, log)
        
        # 2) Update the within-unit variation encoder
        log.info('----------------------------------------------------------------------------------------')
        log.info('        %dth within-unit variation encoder update' % nrepeat)
        log.info('----------------------------------------------------------------------------------------')
        training(argdict_within_unit, log, nrepeat=nrepeat)
        log.info('----------------------------------------------------------------------------------------')
    
    # Fine-tune
    log.info('----------------------------------------------------------------------------------------')
    log.info(' Fine-tuning')
    log.info('----------------------------------------------------------------------------------------')
    training(argdict_total, log, nrepeat=nrepeat)
    
    log.info('----------------------------------------------------------------------------------------')
    log.info('Finished!')
    log.info('----------------------------------------------------------------------------------------')
    # sys.exit()