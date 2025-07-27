# src/utils/loss.py
import tensorflow as tf

def charbonnier_loss(eps=1e-3):
    def loss_fn(y_true, y_pred):
        diff = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(diff * diff + eps * eps))
    return loss_fn

def l1_loss():
    return tf.keras.losses.MeanAbsoluteError()
