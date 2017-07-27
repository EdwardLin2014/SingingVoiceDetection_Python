#########################################################################
## Step 0 - Import Library
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

H5DirStr = '../../../Code/03_Database/iKala/SM_HDF5/'
ModelDirStr = '/Users/EdwardLin/Dropbox/PythonCode/04_iKalaProject/09_SingingDetector/model_20170726_2203'
numTFbins = 42780
numFrames = 115
numBins = 372

#########################################################################
## Define 
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad,
                        gen_nn_ops._relu_grad(grad, op.outputs[0]),
                        tf.zeros_like(grad))
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
## Step 1 - Restore the model
tic = time.time()
sess = tf.Session() 
saver = tf.train.import_meta_graph(ModelDirStr+'/model-33.meta')
saver.restore(sess,tf.train.latest_checkpoint(ModelDirStr))
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("xInput:0")
y_ = graph.get_tensor_by_name("yInput:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
output = graph.get_tensor_by_name("ouputLayer/activation:0")
y = tf.nn.sigmoid(output)
toc = time.time() - tic
print('Restore the model needs %.2f sec' % toc)


#########################################################################
## Step 2 - Calculate Saliency Map
# Change all Relu in the model to GuidedRelu
g = tf.get_default_graph()
g.gradient_override_map({'Relu': 'GuidedRelu'})
# Setup formula for Saliency Map - partial derivative of output w.r.t to input
dydx = tf.gradients(y, x)[0]

#########################################################################
## Step 3 - Calculate the Saliency Map given an input
H5_1FileName = H5DirStr + 'Test_1Song_1024_1024_315.h5'
h5fTest = h5py.File(H5_1FileName, 'r')
mX = h5fTest['mX']
OrigIm = np.transpose(h5fTest['mXImg'])

SalMap = np.zeros((372,2101))
for i in range(2101):    
    tmpSalMap = sess.run(dydx, feed_dict={x:mX[i:i+1,:], keep_prob:1.0})
    tmpSalMap = np.transpose(np.reshape(tmpSalMap,(numFrames,numBins)))
    SalMap[:,i] = tmpSalMap[:,58]
# Plot All Images
threshold = 0.001
plt.figure(figsize=(10, 20), facecolor='w')
plt.subplot(2, 2, 1)
plt.title('input')
plt.imshow(OrigIm)
plt.gca().invert_yaxis()
plt.subplot(2, 2, 2)
plt.title('abs. saliency')
absMap = np.abs(SalMap)
absMap[absMap>threshold] = 1
plt.imshow(absMap)
plt.gca().invert_yaxis()
plt.subplot(2, 2, 3)
plt.title('pos. saliency')
posMap = (np.maximum(0, SalMap) / SalMap.max())
posMap[posMap>threshold] = 1
plt.imshow(posMap)
plt.gca().invert_yaxis()
plt.subplot(2, 2, 4)
plt.title('neg. saliency')
negMap = (np.maximum(0, -SalMap) / -SalMap.min())
negMap[negMap>threshold] = 1
plt.imshow(negMap)
plt.gca().invert_yaxis()
plt.show()
