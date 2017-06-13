import os, sys
import time
import numpy as np
#from scipy.signal import get_window

#########################################################################
## Step 0 - Import Library
Tool_UtilFunc_DirStr = '../../00_Tools/UtilFunc-1.0/';
Tool_BSS_DirStr = '../../00_Tools/bss_eval-3.0/';
WavDirStr = '../../../Code/03_Database/iKala/Wavfile/';
PitchDirStr = '../../../Code/03_Database/iKala/PitchLabel/';
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_BSS_DirStr))
#import UnitTranslation as UT
import Database as DB
import Audio as AD
#from structDT import Param
from structDT import Signal
from bss_eval import bss_eval_1source
#########################################################################
## Step 0 - Obtain Audio File Name
WavFileNames = DB.iKalaWavFileNames(WavDirStr);
numMusics = len(WavFileNames);
#########################################################################
## Step 0 - Obtain Pitch/Voice Label
PitchFileNames = DB.iKalaPitchLabelFileNames(PitchDirStr);
PitchMask = DB.iKalaPitchMask(PitchFileNames,numMusics);
numSamples = 1323008;

trainMusic = [3,4,5,6,7,8,9,10,11,13,14,16,17,20,21,23,27,29,30,33,34,35,36,37,38,39,41,44,46,50,52,53,54,55,56,57,59,60,61,63,64,65,66,67,70,71,73,77,78,82,84,85,86,88,91,92,95,96,97,98,102,103,108,109,114,115,116,117,119,123,124,126,127,128,130,133,141,142,144,145,146,147,149,150,151,153,154,155,156,158,159,160,162,163,164,165,169,170,172,173,174,176,180,184,185,186,187,188,189,190,191,193,194,196,197,199,200,201,203,207,208,209,210,211,215,216,217,219,221,224,225,226,227,230,231,232,233,234,235,236,237,238,239,240,241,242,246,248,249,250,251,252];
valMusic = [1,22,28,32,42,48,49,58,62,72,74,80,83,89,90,93,101,106,120,121,122,125,129,131,136,140,143,148,157,161,168,178,181,182,183,192,195,198,205,206,212,213,214,220,222,229,243,244,245,247];
testMusic = [2,12,15,18,19,24,25,26,31,40,43,45,47,51,68,69,75,76,79,81,87,94,99,100,104,105,107,110,111,112,113,118,132,134,135,137,138,139,152,166,167,171,175,177,179,202,204,218,223,228];
BSS = np.zeros((numMusics,3));

for t in np.arange(numMusics):
    #########################################################################
    ## Step 1 - Import Audio and Create Power Spectrogram
    
    tic = time.time()
    x, fs = AD.audioread(WavFileNames[t])
    Mix = Signal()
    Mix.x = x[:,0]+x[:,1]
    toc = time.time() - tic
    if t < 137:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) );
    else:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) );

    #########################################################################
    ## Step 2 - Obtain singing voice from human-labeled ground truth
    tic = time.time()
    step = int(np.round(fs*30/937));
    mask = np.zeros((numSamples,), dtype=np.int);
    for i in np.arange(937):
        startIdx = (i-1)*step
        if i != 936:
            endIdx = i*step-1
        else:
            endIdx = 1323007
        mask[startIdx:endIdx] = PitchMask[t,i]
    Voice = Signal()
    Voice.y = Mix.x * mask
    toc = time.time() - tic
    print('Obtain singing voice from human-labeled ground truth needs %.2f  sec' % toc);
    
    #########################################################################
    ## Step 3 - BSS Evaluation - Must be carried out in Matlab as 
    ## In Matlab A\B, In Python (numpy.linalg.inv(A)).dot(B)
    ## They are not equal!!!! Becuase of the pseudo-matrix-inverse!
    tic = time.time()
    trueVoice = x[:,1];
    estimatedVoice = Voice.y;
    SDR, SIR, SAR = bss_eval_1source(estimatedVoice, trueVoice);
    BSS[t,0] = SDR;
    BSS[t,1] = SIR;
    BSS[t,2] = SAR;
    print('SDR:%.4f' % SDR);
    print('SIR:%.4f' % SIR);
    print('SAR:%.4f' % SAR);
    toc = time.time() - tic    
    print('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec' % (t, toc));
    