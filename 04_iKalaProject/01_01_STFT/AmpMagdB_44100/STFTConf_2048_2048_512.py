import os, sys
import time
import numpy as np
import scipy.signal as sg

#########################################################################
## Step 0 - Import Library
Tool_UtilFunc_DirStr = '../../../00_Tools/UtilFunc-1.0/';
Tool_SineModel_DirStr = '../../../00_Tools/SineModel-1.0/';
WavDirStr = '../../../../Code/03_Database/iKala/Wavfile/';
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_SineModel_DirStr))
import UnitTranslation as UT
import Database as DB
import Audio as AD
from structDT import Param
from structDT import Signal
import SineModel as SM

#########################################################################
## Step 0 - Parmaters Setting
# STFT
Parm = Param()
Parm.M = 2048;                                      # Window Size, 46.44ms
Parm.window = sg.get_window('hann', Parm.M);        # Window in Vector Form
Parm.N = 2048;                                      # Analysis DFT Size, 46.44ms
Parm.H = 512;                                       # Hop Size, 11.61ms
Parm.fs = 44100;                                    # Sampling Rate, 44.10K Hz
Parm.t = 1;                                         # Need All Peaks, in term of Mag Level

#########################################################################
## Step 0 - Obtain Audio File Name
WavFileNames = DB.iKalaWavFileNames(WavDirStr);
numMusics = len(WavFileNames);
numSamples = 1323008;
# Statistics - Magnitude
MixAmp = np.zeros((numMusics,2))
OverallMag = np.zeros((numMusics,2))
OverallMagdB = np.zeros((numMusics,2))

for t in np.arange(numMusics):
    #########################################################################
    ## Step 1 - Import Audio and Create Power Spectrogram
    
    tic = time.time()
    x, fs = AD.audioread(WavFileNames[t])
    Mix = Signal()
    Mix.x = x[:,0]+x[:,1]
    MixAmp[t,0] = min(Mix.x)
    MixAmp[t,1] = max(Mix.x)
    # Spectrogram Dimension - Parm.numBins:1025 X Parm.numFrames:2584 = 2,648,600
    Mix.X, Mix.mX, Mix.pX, Parm.remain, Parm.numFrames, Parm.numBins = SM.stft(Mix.x, Parm);    
    OverallMag[t,0] = min(Mix.mX.min(0))
    OverallMag[t,1] = max(Mix.mX.max(0))
    OverallMagdB[t,0] = UT.MagTodB(min(Mix.mX.min(0)))
    OverallMagdB[t,1] = UT.MagTodB(max(Mix.mX.max(0)))
    toc = time.time() - tic
    if t < 137:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) );
    else:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) );