import tensorflow as tf
import numpy as np
import os
from classifiers.squeezenet import SqueezeNet

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def load_model(sess):
    SAVE_PATH = 'datasets/squeezenet.ckpt'
    #if not os.path.exists(SAVE_PATH):
    #    raise ValueError("SqueezeNet is not downloaded.")
    return SqueezeNet(save_path=SAVE_PATH, sess=sess)


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
