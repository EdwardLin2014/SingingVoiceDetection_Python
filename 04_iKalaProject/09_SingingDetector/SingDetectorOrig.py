#########################################################################
## Step 0 - Import Library
import os, sys
import time
import h5py

currTime = time.strftime("%Y%m%d_%H%M")
Tool_DNN_DirStr = '../../00_Tools/MyDNN-1.0/'
H5DirStr = '../../../Code/03_Database/iKala/SM_HDF5/'
ModelDirStr = '/Users/EdwardLin/Dropbox/PythonCode/04_iKalaProject/09_SingingDetector/model_' + currTime
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_DNN_DirStr))
import MyDNN as MD
import tensorflow as tf

numTFbins = 42780
numFrames = 115
numBins = 372
numClass = 1
MaxEpochs = 50
numTrain = 152
numValid = 50
numTest = 50
    
#########################################################################
## Step 0 - Obtain dataset
totaltic = time.time()

print('Start to obtain dataset. Please wait......')
H5FileName = H5DirStr + 'Spec_1024_1024_315.h5'

tic = time.time()
h5f = h5py.File(H5FileName, 'r')
trainSet = h5f['train']
trainLabel = h5f['trainLabel']
toc = time.time() - tic
print('Obtained Training set need %.2f sec' % toc)

tic = time.time()
valSet = h5f['valid']
valLabel = h5f['validLabel']
toc = time.time() - tic
print('Obtained validation set need %.2f sec' % toc)

tic = time.time()
testSet = h5f['test']
testLabel = h5f['testLabel']
toc = time.time() - tic
print('Obtained Test set need %.2f sec' % toc)

totaltoc = time.time() - totaltic
print('Obtain all dataset needs %.2f sec' % totaltoc)

#########################################################################
## Step 1 - Create ConvNet
tic = time.time()
sess = tf.InteractiveSession()
# Input and Ouput Layer & placeholder
x = tf.placeholder(tf.float32, [None, numTFbins], name='xInput')
y_ = tf.placeholder(tf.float32, [None, 1], name='yInput')
x_image = tf.reshape(x, [-1,numFrames,numBins,1])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# conv & pool layer - activation='relu' - tf.contrib.layers.xavier_initializer 
Conv1 = MD.conv2d_layer(x_image, 3,3,1,64,'Conv1')
Conv2 = MD.conv2d_layer(Conv1,3,3,64,32,'Conv2')
MaxPool = MD.max_pool_3x3(Conv2)
Conv3 = MD.conv2d_layer(MaxPool,3,3,32,128,'Conv3')
Conv4 = MD.conv2d_layer(Conv3,3,3,128,64,'Conv4')
# fully-connected layer + dropout
Conv4_flat = tf.reshape(Conv4, [-1, 39 * 124 * 64])
fc1_drop = tf.nn.dropout(Conv4_flat, keep_prob, name='drop1')
fc1 = MD.nn_layer(fc1_drop, 39 * 124 * 64, 256, 'fullyConnect1')
fc2_drop = tf.nn.dropout(fc1, keep_prob, name='drop2')
fc2 = MD.nn_layer(fc2_drop, 256, 64, 'fullyConnect2')
# output layer
y = MD.nn_layer(fc2, 64, 1, 'ouputLayer', act=tf.identity)
# Define loss and optimizer
crossEntropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
tf.global_variables_initializer().run()
toc = time.time() - tic
print('Create CNN Model needs %.2f sec' % toc)

#########################################################################
## Step 2 - Train the CNN and Write the Log
totalic = time.time()
saver = tf.train.Saver()
for i in range(MaxEpochs):
    # Training
    tic = time.time()
    for t in range(numTrain):
        StartIdx = t*191
        EndIdx = StartIdx+95
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx], keep_prob: 0.5})
        StartIdx = EndIdx
        EndIdx = (t+1)*191
        sess.run(train_step, feed_dict={x:trainSet[StartIdx:EndIdx,:], y_:trainLabel[StartIdx:EndIdx], keep_prob: 0.5})
    toc = time.time() - tic
    print('%dth Train epoch; times need %.2f sec' % (i+1,toc))

    ## Step 3 - Save the Trained Model
    stic = time.time()
    if tf.gfile.Exists(ModelDirStr):
        tf.gfile.DeleteRecursively(ModelDirStr)
    tf.gfile.MakeDirs(ModelDirStr)
    saver.save(sess, ModelDirStr+'/model', global_step=i)
    stoc = time.time() - stic
    print('Save the %dth Trained Model needs %.2f sec' % (i,stoc))

totaloc = time.time() - totalic
print('Total Training Time needs %.2f sec' % totaloc)
