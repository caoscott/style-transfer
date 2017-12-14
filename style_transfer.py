import numpy as np
import argparse
import tensorflow as tf
from classifiers.squeezenet import SqueezeNet
from utils import load_model
from image_utils import load_image, preprocess_image, deprocess_image
import matplotlib.pyplot as plt


def rel_error(x, y):
    return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
        
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
                                
    Returns:
    - scalar content loss
    """
    l2_loss = tf.nn.l2_loss(tf.subtract(content_current, content_original))
    return tf.multiply(2*content_weight, l2_loss)


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
        
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for a single image.    
    - normalize: optional, whether to normalize the Gram matrix. If True, divide the Gram matrix by the number of neurons (H * W * C)
                                          
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized). Gram matrices for the input image.
    """
    shape = tf.shape(features)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    features_reshaped = tf.reshape(features, (H * W, N * C))
    gram = tf.matmul(features_reshaped, features_reshaped, transpose_a = True)
    if normalize:
        gram = tf.divide(gram, tf.cast(H * W * C, tf.float32))
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
        
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is a Tensor giving the Gram matrix the source style image computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i] is a scalar giving the weight for the style loss at layer style_layers[i].
    
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    style_loss = 0
    for i, style_layer in enumerate(style_layers):
        current_gram = gram_matrix(feats[style_layer])
        diff = tf.subtract(current_gram, style_targets[i])
        style_loss += 2 * style_weights[i] * tf.nn.l2_loss(diff)
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
        
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
                            
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss for img weighted by tv_weight.
    """
    cropped = img[:, :-1, :-1, :]
    u_shift = img[:, 1:, :-1, :]
    r_shift = img[:, :-1, 1:, :]
    u_loss = tf.nn.l2_loss(u_shift - cropped)
    r_loss = tf.nn.l2_loss(r_shift - cropped)
    return 2 * tv_weight * (u_loss + r_loss)


def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight, style_layers, style_weights, tv_weight, init_random = False):
    """Run style transfer!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = model.extract_features(model.image)
    content_target = sess.run(feats[content_layer], {model.image: content_img[None]})

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    style_feat_vars = [feats[idx] for idx in style_layers]
    style_target_vars = []
    # Compute list of TensorFlow Gram matrices
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
    style_targets = sess.run(style_target_vars, {model.image: style_img[None]})

    # Initialize generated image to content image
    
    if init_random:
        img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
    else:
        img_var = tf.Variable(content_img[None], name="image")

    # Extract features on generated image
    feats = model.extract_features(img_var)
    # Compute loss
    c_loss = content_loss(content_weight, feats[content_layer], content_target)
    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    t_loss = tv_loss(img_var, tv_weight)
    loss = c_loss + s_loss + t_loss
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = 200

    # Create and initialize the Adam optimizer
    lr_var = tf.Variable(initial_lr, name="lr")
    # Create train_op that updates the generated image when run
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
    # Initialize the generated image and optimization variables
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))
    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))
    
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess_image(content_img))
    axarr[1].imshow(deprocess_image(style_img))
    plt.savefig('orig_imgs')
    plt.show()
    plt.figure()
    
    # Hardcoded handcrafted 
    for t in range(max_iter):
        # Take an optimization step to update img_var
        sess.run(train_op)
        if t < decay_lr_at:
            sess.run(clamp_image_op)
        if t == decay_lr_at:
            sess.run(tf.assign(lr_var, decayed_lr))
        if t % 100 == 0:
            print('Iteration {}'.format(t))
            img = sess.run(img_var)
            plt.imshow(deprocess_image(img[0], rescale=True))
            plt.axis('off')
            plt.savefig('iteration{}'.format(t))
            plt.show()
    print('Iteration {}'.format(t))
    img = sess.run(img_var)        
    plt.imshow(deprocess_image(img[0], rescale=True))
    plt.axis('off')
    plt.savefig('style_transfer.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Style transfer.')
    parser.add_argument('-c', '--content-image', type=str)
    parser.add_argument('-s', '--style-image', type=str)
    parser.add_argument('--image-size', type=int, default=192)
    parser.add_argument('--style-size', type=int, default=512)
    parser.add_argument('--content-layer', type=int, default=3)
    parser.add_argument('--content-weight', type=float, default=5e-2)
    parser.add_argument('--style-layers', type=tuple, default=(1, 4, 6, 7))
    parser.add_argument('--style-weights', type=tuple, default=(20000, 500, 12, 1))
    parser.add_argument('--tv-weight', type=float, default=5e-2)
    args = parser.parse_args()

    tf.reset_default_graph()
    sess = get_session()
    model = load_model(sess)
    
    # Composition VII + Tubingen
    params1 = {
        'content_image' : args.content_image,
        'style_image' : args.style_image,
        'image_size' : args.image_size,
        'style_size' : args.style_size,
        'content_layer' : args.content_layer,
        'content_weight' : args.content_weight, 
        'style_layers' : args.style_layers,
        'style_weights' : args.style_weights,
        'tv_weight' : args.tv_weight
    }

    style_transfer(**params1)
