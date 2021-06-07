import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Dropout
from keras.layers import UpSampling2D, AveragePooling2D, Add, Concatenate
from keras.layers import Layer, Input, Lambda
from keras.initializers import TruncatedNormal, RandomNormal, he_normal

# import keras.backend as k
# import tensorflow as tf
####################################################################################################################################

def conv2D_block(x, name, filters, kernel_size,
                 strides=1, padding='same',
                 kernel_initializer=TruncatedNormal(stddev=0.0099999),
                 bn=True, epsilon=1e-5, momentum=0.9, activation='relu', dropout=0.):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation'%name)(x)
    if dropout > 0.: x = Dropout(dropout, name='%s_dropout'%name)(x)
    return x

def conv2Dtranspose_block(x, name, filters, kernel_size, 
                          strides=1, padding='same',
                          kernel_initializer=TruncatedNormal(stddev=0.0099999),
                          bn=True, epsilon=1e-5, momentum=0.9, activation='relu', dropout=0.):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation'%name)(x)
    if dropout > 0.: x = Dropout(dropout, name='%s_dropout'%name)(x)
    return x

def dense_block(x, name, units, 
                kernel_initializer=RandomNormal(stddev=0.0099999),
#                 kernel_initializer=he_normal(),
                bn=True, epsilon=1e-5, momentum=0.9, activation='relu', dropout=0.):
    x = Dense(units=units, kernel_initializer=kernel_initializer, name='%s_dense'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation'%name)(x)
    if dropout > 0.: x = Dropout(dropout, name='%s_dropout'%name)(x)
    return x

def res_conv2D_up_block(input_x, name, filters, 
                        padding='same', kernel_initializer=TruncatedNormal(stddev=0.0099999),
                        bn=True, epsilon=1e-5, momentum=0.9, activation='relu'):

    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_1'%name)(input_x)
    else: x = input_x
    if activation != None: x = Activation(activation, name='%s_activation_1'%name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_1'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_2'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_2'%name)(x)
    x = UpSampling2D(size=2, name='%s_upsampling_1' % name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_2'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_3'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_3'%name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_3'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_4'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_4'%name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_4'%name)(x)


    x_2 = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_5'%name)(input_x)
    x_2 = UpSampling2D(size=2, name='%s_upsampling_2' % name)(x_2)

    x = Add(name='%s_skip' % name)([x, x_2])
    
    return x

def res_conv2D_down_block(input_x, name, filters, 
                        padding='same', kernel_initializer=TruncatedNormal(stddev=0.0099999),
                        bn=True, epsilon=1e-5, momentum=0.9, activation='relu'):

    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_1'%name)(input_x)
    else: x = input_x
    if activation != None: x = Activation(activation, name='%s_activation_1'%name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_1'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_2'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_2'%name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_2'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_3'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_3'%name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_3'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_4'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_4'%name)(x)
    x = AveragePooling2D(pool_size=2, padding=padding, name='%s_avgpool_1' % name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_4'%name)(x)
    
    x_2 = AveragePooling2D(pool_size=2, padding=padding, name='%s_avgpool_2' % name)(input_x)
    x_2 = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_5'%name)(x_2)
    x = Add(name='%s_skip' % name)([x, x_2])
    
    return x

def res_conv2D_block(input_x, name, filters, 
                     padding='same', kernel_initializer=TruncatedNormal(stddev=0.0099999),
                     bn=True, epsilon=1e-5, momentum=0.9, activation='relu'):

    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_1'%name)(input_x)
    else: x = input_x
    if activation != None: x = Activation(activation, name='%s_activation_1'%name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_1'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_2'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_2'%name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_2'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_3'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_3'%name)(x)
    x = Conv2D(filters=filters*2, kernel_size=3, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_3'%name)(x)
    if bn: x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='%s_bn_4'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_4'%name)(x)
    x = Conv2D(filters=filters, kernel_size=1, padding=padding,
                kernel_initializer=kernel_initializer, name='%s_conv_4'%name)(x)
    
    x = Add(name='%s_skip' % name)([x, input_x])
    
    return x

def res_dense_block(input_x, name, units, 
                    kernel_initializer=TruncatedNormal(stddev=0.0099999),
                    bn=True, epsilon=1e-5, momentum=0.9, activation='relu'):

    if activation != None: x = Activation(activation, name='%s_activation_1'%name)(input_x)
    else: x = input_x
    x = Dense(units=units, kernel_initializer=kernel_initializer, name='%s_dense_1'%name)(x)
    if activation != None: x = Activation(activation, name='%s_activation_2'%name)(x)
    x = Dense(units=units, kernel_initializer=kernel_initializer, name='%s_dense_2'%name)(x)
    
    x_2 = Dense(units=units, kernel_initializer=kernel_initializer, name='%s_dense_3'%name)(input_x)

    x = Add(name='%s_skip' % name)([x, x_2])
    
    return x


####################################################################################################################################

def get_compute_blurriness_model(image_shape):
    def _init_laplace_filter(shape, dtype=None):
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).reshape([3,3,1,1])
    def set_rgb_to_gray(x):
        import tensorflow as tf
        return tf.image.rgb_to_grayscale(x)
    def get_variance(x):
        import keras.backend as k
        return k.var(x, axis=(1,2,3))
    image_input = Input(shape=image_shape, name='image_input', dtype='float32')
    if image_shape[-1] > 1: # RGB
        image = Lambda(set_rgb_to_gray, name='rgb_to_gray')(image_input)
    else: image=image_input
    laplace_transform = Conv2D(filters=1, kernel_size=3,
                               kernel_initializer= _init_laplace_filter, bias_initializer='zeros', padding='same')(image)
    laplace_var = Lambda(get_variance, name='variance')(laplace_transform)
    return Model(image_input, laplace_var, name='blurriness')

####################################################################################################################################

def get_class_mean_by_class_index(latent_variable, class_info):
    import tensorflow as tf
    unique_class, class_idx, class_count = tf.unique_with_counts(class_info[:,0])
    n_label = tf.shape(unique_class)
    shape = tf.concat([n_label, tf.shape(latent_variable)[1:]], axis=0)
    num_points_in_class = tf.broadcast_to(tf.cast(class_count, 'float32')[:,tf.newaxis], shape=shape)
    centroids = tf.scatter_nd(class_idx[:, tf.newaxis], latent_variable, shape=shape) / num_points_in_class
    centroids_spread = tf.gather(centroids, class_idx)
    return centroids_spread, centroids, unique_class, class_count, n_label[0]

####################################################################################################################################

def get_b(args):
    latent_variable, class_info = args
    sample_b, b_given_x, _, _, _ = get_class_mean_by_class_index(latent_variable, class_info)
    return [sample_b, b_given_x]
    
def get_b_estimation_var(arg):
    import keras.backend as k
    import tensorflow as tf
    b_j_given_x_j, sample_b, cls_info = arg
    base_size_input = k.zeros_like(b_j_given_x_j[:,0])
#         sq = k.mean(tf.square(b_j_given_x_j - sample_b), axis=1, keepdims=True)
    sq = tf.square(b_j_given_x_j - sample_b)
    _, class_mean, _, _, _ = get_class_mean_by_class_index(sq, cls_info)
    stat = tf.reduce_mean(class_mean, axis=0)
    stat = tf.reduce_mean(stat)
    return stat + base_size_input

def get_pairwise_distance(arg):
    # ref: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    import keras.backend as k
    import tensorflow as tf
    A = arg[0]
    base_size_input = arg[1]
    r = tf.reduce_sum(A*A, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return -0.0001 * tf.reduce_mean(D) + k.zeros_like(base_size_input[:,0])

def get_b_penalty_loss(arg, sigma=10., zdim=8, kernel='IMQ', p_z='normal'):
     ### TODO : fix...ugly...
    import keras.backend as k
    import tensorflow as tf
#     from .ops import mmd_penalty
    prior_b, b_given_x = arg
    nsize = tf.shape(b_given_x)[0]
    sample_pb = prior_b[:nsize]
    base_size_input = k.zeros_like(prior_b[:,0])
    stat = mmd_penalty(sample_pb, b_given_x, sigma, nsize, kernel, p_z, zdim)
    return stat + base_size_input

def mean_reconstruction_l1_loss(args):
    import keras.backend as k
    import tensorflow as tf
    y_true, y_pred = args
    loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.02 * tf.reduce_mean(loss)
    return loss + k.zeros_like(dist)

def mean_reconstruction_l2sq_loss(args):
    import keras.backend as k
    import tensorflow as tf
    y_true, y_pred = args
    dist = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    loss = dist
    loss = 0.05 * tf.reduce_mean(loss)
    return loss + k.zeros_like(dist)

def mean_reconstruction_l2sq_loss_e(args):
    import keras.backend as k
    import tensorflow as tf
    y_true, y_pred, cls_info = args
    dist = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
#     loss = dist
    _, loss, _, _, _ = get_class_mean_by_class_index(dist[:,tf.newaxis], cls_info)
    loss = 0.05 * tf.reduce_mean(loss)
    return loss + k.zeros_like(dist)

####################################################################################################################################

def sampling(args, z_dim):
    import keras.backend as k
    mean, var = args
    # scale = k.sqrt(k.exp(k.clip(log_var, -100., 100.)))
    scale = k.clip(var, 1e-16, 1000.)
    batch_size = k.shape(mean)[0]
    epsilon = k.random_normal(shape=(batch_size, z_dim), mean=0., stddev = 1., dtype='float32')
    return mean + scale * epsilon

def concat_with_uniform_sample(x, z_dim=8):
    import keras.backend as k
    batch_size = k.shape(x)[0]
    uniform_sample = k.random_uniform(shape=(batch_size, z_dim), minval=0.0, maxval=1.0)
    return k.concatenate([x, uniform_sample], axis=1)

#########################################################################################################################

def get_qz_trick_loss(q_z):
    # Non-saturating loss trick
    import keras.backend as k
    import tensorflow as tf
    loss_qz_trick = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_z, labels=k.ones_like(q_z)))
#     loss_qz_trick = - tf.reduce_mean(tf.log(q_z))
    return loss_qz_trick + k.zeros_like(q_z[:,0])

def get_qz_trick_with_weight_loss(args):
    # Non-saturating loss trick
    import keras.backend as k
    import tensorflow as tf
    q_z = args[0]
    cls_info = args[1]
    qz_trick = tf.nn.sigmoid_cross_entropy_with_logits(logits=q_z, labels=k.ones_like(q_z))
#     qz_trick = tf.log(q_z)
    _, loss_qz_trick, _, _, _ = get_class_mean_by_class_index(qz_trick, cls_info)
    loss_qz_trick = tf.reduce_mean(loss_qz_trick)
#     loss_qz_trick = - tf.reduce_mean(loss_qz_trick)
    return loss_qz_trick + k.zeros_like(q_z[:,0])
    
def mmd_penalty(sample_pz, sample_qz, sigma, n, kernel='IMQ', p_z='normal', z_dim=8):
    import tensorflow as tf
    sigma2_p = sigma ** 2
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
#     half_size = (n * n - n) / 2
    half_size = tf.cast((n * n - n) / 2, tf.int32)

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    if kernel == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
#         sigma2_k = z_dim* sigma2_p
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif kernel == 'IMQ':
        ## k(x, y) = C / (C + ||x - y||^2)
        # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        if p_z == 'normal':
            Cbase = 2. * z_dim * sigma2_p
        elif p_z == 'sphere':
            Cbase = 2.
        elif p_z == 'uniform':
            # E ||x - y||^2 = E[sum (xi - yi)^2]
            #               = zdim E[(xi - yi)^2]
            #               = const * zdim
            Cbase = z_dim
        stat = 0.
#         for scale in [0.01, 0.02, 0.05, .1, .2, .5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.]:
        for scale in [.1, .2, .5, 1., 2., 5., 10., 50., 100.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    stat = tf.where(tf.is_nan(stat), tf.zeros_like(stat), stat)
    return stat

def get_entropy_loss_with_logits(logits):
    import keras.backend as k
    import tensorflow as tf
    conditional_entropy = - tf.reduce_mean(tf.nn.softmax(logits, axis=1) * tf.nn.log_softmax(logits, axis=1))
#     phat = tf.nn.softmax(logits, axis=-1)
#     conditional_entropy = - tf.reduce_mean(tf.reduce_sum(phat * tf.log(tf.clip(phat)), axis=-1))
    return conditional_entropy + k.zeros_like(logits[:,0])

def get_hsic(x):
    """
    Refers to original Tensorflow implementation: https://github.com/romain-lopez/HCV
    Refers to original implementations
        - https://github.com/kacperChwialkowski/HSIC
        - https://cran.r-project.org/web/packages/dHSIC/index.html
    """
    from scipy.special import gamma
    import tensorflow as tf
    import numpy as np

    def bandwidth(d):
        """
        in the case of Gaussian random variables and the use of a RBF kernel, 
        this can be used to select the bandwidth according to the median heuristic
        """
        gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
        return 1. / (2. * gz**2)

    def knl(x1, x2, gamma=1.): 
        dist_table = tf.expand_dims(x1, 0) - tf.expand_dims(x2, 1)
        return tf.transpose(tf.exp(-gamma * tf.reduce_sum(dist_table **2, axis=2)))

#     def hsic(z, s):
    # use a gaussian RBF for every variable
    
    z, s = x[0], x[1]
     
    d_z = z.get_shape().as_list()[1]
    d_s = s.get_shape().as_list()[1]

    zz = knl(z, z, gamma= bandwidth(d_z))
    ss = knl(s, s, gamma= bandwidth(d_s))

    hsic = 0
    hsic += tf.reduce_mean(zz * ss) 
    hsic += tf.reduce_mean(zz) * tf.reduce_mean(ss)
    hsic -= 2 * tf.reduce_mean( tf.reduce_mean(zz, axis=1) * tf.reduce_mean(ss, axis=1) )
    
    stat = tf.sqrt(tf.clip_by_value(hsic, 1e-16, hsic))
#     stat = hsic
#     stat = tf.where(tf.is_nan(stat), tf.zeros_like(stat), stat)
    return stat + tf.zeros_like(z[:,0])

####################################################################################################################################

def get_batch_variance(x):
    import tensorflow as tf
    import keras.backend as k
    sample_axis=0
    x -= tf.reduce_mean(x, axis=sample_axis, keepdims=True)
    n_samples = tf.cast(tf.shape(x)[0], x.dtype)
    cov = tf.matmul(x, x, transpose_a=True) / (n_samples)
    return cov + 1e-8 * tf.eye(tf.shape(cov)[0], dtype=x.dtype)

def get_mutual_information_from_gaussian_sample(args):
    import tensorflow as tf
    e, b, z = args
    cov_e = get_batch_variance(tf.cast(e, tf.float64))
    cov_b = get_batch_variance(tf.cast(b, tf.float64))
    cov_z = get_batch_variance(tf.cast(z, tf.float64))
    stat = 0.5 * (tf.linalg.logdet(cov_e) + tf.linalg.logdet(cov_b) - tf.linalg.logdet(cov_z))
    return tf.cast(stat, tf.float32) + tf.zeros_like(e[:,0])

def get_batch_covariance(x, y):
    import tensorflow as tf
    import keras.backend as k
    sample_axis=0
    x -= tf.reduce_mean(x, axis=sample_axis, keepdims=True)
    y -= tf.reduce_mean(y, axis=sample_axis, keepdims=True)
    n_samples = tf.cast(tf.shape(x)[0], x.dtype)
    cov = tf.matmul(x, y, transpose_a=True) / (n_samples)
    return cov + 1e-6 * 1e-8 * tf.eye(tf.shape(cov)[0], dtype=x.dtype)

# def get_mutual_information_from_gaussian_sample(args):
#     import tensorflow as tf
#     import keras.backend as k
#     e, b, z = args
#     b_64 = tf.cast(b, tf.float64)
#     e_64 = tf.cast(e, tf.float64)
#     sigma_b_sq = k.var(b_64) ** 2.
#     sigma_e_sq = k.var(e_64) ** 2.
#     cov_be = get_batch_covariance(b_64, e_64)
#     schur = sigma_b_sq - tf.matmul(cov_be, cov_be, transpose_b=True) / sigma_e_sq
#     stat = 0.5 * (tf.shape(b_64)[1] tf.log(sigma_b_sq) - tf.linalg.logdet(schur))
#     return tf.cast(stat, tf.float32) + tf.zeros_like(e[:,0])

def zeros_like_layer(x):
    import keras.backend as k
    return k.zeros_like(x[:,0])

####################################################################################################################################

class SelfAttention2D(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernelf = self.add_weight(name='convf', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]//8),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelg = self.add_weight(name='convg', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]//8),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelh = self.add_weight(name='convh', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        super(SelfAttention2D, self).build(input_shape)

    def call(self, x):
        import keras.backend as k
        import tensorflow as tf
        def hw_flatten(x): return tf.reshape(x,[-1, k.shape(x)[1]*k.shape(x)[2], k.shape(x)[3]])
        
        f = k.conv2d(x, kernel=self.kernelf, padding='same') # [bs, h, w, c']
        g = k.conv2d(x, kernel=self.kernelg, padding='same') # [bs, h, w, c']
        h = k.conv2d(x, kernel=self.kernelh, padding='same') # [bs, h, w, c]
        
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N] where N = h * w
        beta = k.softmax(s)  # attention map [bs, N, N]
    
        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, c]
        o = tf.reshape(o, [-1, k.shape(x)[1], k.shape(x)[2], k.shape(x)[3]])  # [bs, h, w, c]
        x = self.gamma * o + x
        return [x, beta]

    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0], input_shape[1]*input_shape[2], input_shape[1]*input_shape[2])]