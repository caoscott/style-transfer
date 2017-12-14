import os
import tensorflow as tf
import numpy as np
from utils import get_session, rel_error, load_model
from image_utils import preprocess_image, load_image
from style_transfer import content_loss, gram_matrix, style_loss, tv_loss

def content_loss_test(correct):
    content_layer = 3
    content_weight = 6e-2
    c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img_test})
    bad_img = tf.zeros(content_img_test.shape)
    feats = model.extract_features(bad_img)[content_layer]
    output = sess.run(content_loss(content_weight, c_feats, feats))
    error = rel_error(correct, output)
    print('Maximum error is {}'.format(error))


def gram_matrix_test(correct):
    gram = gram_matrix(model.extract_features()[5])
    output = sess.run(gram, {model.image: style_img_test})
    error = rel_error(correct, output)
    print('Maximum error is {}'.format(error))


def style_loss_test(correct):
    style_layers = [1, 4, 6, 7]
    style_weights = [300000, 1000, 15, 3]

    feats = model.extract_features()
    style_target_vars = [gram_matrix(feats[idx]) for idx in style_layers]
    style_targets = sess.run(style_target_vars, {model.image: style_img_test})

    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    output = sess.run(s_loss, {model.image: content_img_test})
    error = rel_error(correct, output)
    print('Error is {}'.format(error))


def tv_loss_test(correct):
    tv_weight = 2e-2
    t_loss = tv_loss(model.image, tv_weight)
    output = sess.run(t_loss, {model.image: content_img_test})
    error = rel_error(correct, output)
    print('Error is {}'.format(error))


if __name__ == '__main__':
    tf.reset_default_graph() # remove all existing variables in the graph 
    sess = get_session() # start a new Session
    model = load_model(sess)

    content_img_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
    style_img_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
    answers = np.load('style-transfer-checks-tf.npz')

    content_loss_test(answers['cl_out'])

    gram_matrix_test(answers['gm_out'])
    
    style_loss_test(answers['sl_out'])

    tv_loss_test(answers['tv_out'])
