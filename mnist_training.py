import os, sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

sys.path.append('/'.join(os.path.abspath(__file__).split('/')))
from ornstein_auto_encoder import logging_daily
from ornstein_auto_encoder import configuration
from ornstein_auto_encoder.utils import argv_parse
from ornstein_auto_encoder.training import training

if __name__ == '__main__':
    argdict = argv_parse(sys.argv)
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    logger = logging_daily.logging_daily(argdict['log_info'][0])
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)

    log.info('Argument input')
    for argname, arg in argdict.items():
        log.info('    {}:{}'.format(argname,arg))

    training(argdict, log)

    # sys.exit()