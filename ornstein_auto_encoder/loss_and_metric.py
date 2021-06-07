import numpy as np
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as k
from keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
###############################################################################################################################

def gan_loss(y_true, y_pred):
    p_z = y_pred[:,0]
    q_z = y_pred[:,1]
#     # loss_pz = k.mean(k.binary_crossentropy(target=k.ones_like(p_z), 
#     #                                        output=p_z, from_logits=True), axis=-1)
#     # loss_qz = k.mean(k.binary_crossentropy(target=k.zeros_like(q_z), 
#     #                                        output=q_z, from_logits=True), axis=-1)
    
    loss_pz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_z, labels=tf.ones_like(p_z)))
    loss_qz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_z, labels=tf.zeros_like(q_z)))
    
#     # loss_pz = binary_crossentropy(k.ones_like(p_z), p_z)
#     # loss_qz = binary_crossentropy(k.zeros_like(q_z), q_z)
    return loss_pz+loss_qz

def ancillary_loss(y_true, y_pred):
#     labels = y_true
#     logits = y_pred
#     loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name="cross_entropy_with_logit")
    loss = categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0)
    return loss

# def match_penalty_loss(y_true, y_pred):
#     # Non-saturating loss trick
#     # loss_qz_trick = k.mean(k.binary_crossentropy(target=k.ones_like(y_pred), 
#     #                                              output=y_pred, from_logits=True), axis=-1)
    
#     loss_qz_trick = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=tf.ones_like(y_pred)))
    
#     # loss_qz_trick = binary_crossentropy(k.ones_like(y_pred), y_pred)
#     return loss_qz_trick

def mean_penalty_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def first_penalty_loss(y_true, y_pred):
    return y_pred[0]

## TODO : bug... cannot load..
# def random_effect_penalty_loss(sigma=10):
#     def _random_effect_penalty_loss(y_true, y_pred):
#         ## b ~ N_{nsize}(0.,sigma_{nsize})
#         nsize = k.sum(k.ones_like(y_pred))
#         mean_b = tf.reduce_mean(y_pred)
#         mean_loss = tf.square(0. - mean_b)
#         sigma_b = k.sum(tf.square(y_pred - mean_b)) / (nsize - 1.+1e-14)
#         sigma_loss = tf.square(np.square(sigma) - sigma_b)
#         return mean_loss + sigma_loss
#     return _random_effect_penalty_loss

def reconstruction_l2_loss(y_true, y_pred):
    # loss = k.sum(k.square(y_true - y_pred), axis=[1, 2, 3])
    # loss = 0.2 * k.mean(k.sqrt(1e-08 + loss))
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
    return loss

def reconstruction_l1_loss(y_true, y_pred):
    loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.02 * tf.reduce_mean(loss)
    # loss = k.sum(k.abs(y_true - y_pred), axis=[1, 2, 3])
    # loss = 0.02 * k.mean(k.sqrt(1e-08 + loss))
    return loss

def reconstruction_l2sq_loss(y_true, y_pred):
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.05 * tf.reduce_mean(loss)
    # loss = k.sum(k.square(y_true - y_pred), axis=[1, 2, 3])
    # loss = 0.05 * k.mean(loss)
    return loss

def pretrain_loss(y_true, y_pred):
    ### TODO : random effect + noise, random_effect, noise
    
#     # Adding ops to pretrain the encoder so that mean and covariance
#     # of Qz will try to match those of Pz
    nsize = k.sum(k.ones_like(y_true[:,0]))
    pz = y_true
    qz = y_pred
    mean_pz = tf.reduce_mean(pz, axis=0, keep_dims=True)
    mean_qz = tf.reduce_mean(qz, axis=0, keep_dims=True)
    mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
    
    cov_pz = tf.matmul(pz - mean_pz, pz - mean_pz, transpose_a=True)
    cov_pz /= (nsize - 1.+1e-14)
    cov_qz = tf.matmul(qz - mean_qz, qz - mean_qz, transpose_a=True)
    cov_qz /= (nsize - 1.+1e-14)
    cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
    return mean_loss + cov_loss