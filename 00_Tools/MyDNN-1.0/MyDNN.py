# -*- coding: utf-8 -*-
import tensorflow as tf

def feed_dict(dataset,label,x_pl,y_pl,ithSong=-1,numFrames=937,StartIdx=-1,EndIdx=-1):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if ithSong == -1:
        xs = dataset
        ys = label
    elif StartIdx == -1:
        StartIdx = ithSong*numFrames
        EndIdx = (ithSong+1)*numFrames
        xs = dataset[StartIdx:EndIdx,:]
        ys = label[StartIdx:EndIdx,:]
    else:
        xs = dataset[StartIdx:EndIdx,:]
        ys = label[StartIdx:EndIdx,:]

    return {x_pl: xs, y_pl: ys}

def weight_variable(shape,initializer=tf.contrib.layers.xavier_initializer()):
    """Create a weight variable with appropriate initialization."""
    # We can't initialize these variables to 0 - the network will get stuck.
    #initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape=shape))

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            #tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        #tf.summary.histogram('activations', activations)
        
        return activations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_12x12(x):
    return tf.nn.max_pool(x, ksize=[1, 12, 12, 1], strides=[1, 12, 12, 1], padding='SAME')

def max_pool_5x12(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 12, 1], strides=[1, 5, 12, 1], padding='SAME')

def conv2d_layer(input_tensor, patch_width, patch_height, input_channel, output_channel, layer_name, act=tf.nn.relu):
    """
    Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([patch_width, patch_height, input_channel, output_channel])
        with tf.name_scope('biases'):
            biases = bias_variable([output_channel])
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
    return activations
