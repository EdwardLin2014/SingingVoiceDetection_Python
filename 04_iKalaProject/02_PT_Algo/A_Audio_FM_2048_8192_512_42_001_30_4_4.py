import os, sys
import time
import numpy as np
import scipy.signal as sg

#########################################################################
## Step 0 - Import Library
AudioOutDirStr = '../../02_Audio/02_PT_Algo/'
Tool_UtilFunc_DirStr = '../../00_Tools/UtilFunc-1.0/'
Tool_SineModel_DirStr = '../../00_Tools/SineModel-1.0/'
WavDirStr = '../../../Code/03_Database/iKala/Wavfile/'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_SineModel_DirStr))
import UnitTranslation as UT
import Database as DB
import Audio as AD
from structDT import Param
from structDT import Signal
import SineModel as SM
import PartialsUtil as PU
import FormatTransform as FT

#########################################################################
## Step 0 - Parmaters Setting
# STFT
Parm = Param()
Parm.M = 2048;                                      # Window Size, 46.44ms
Parm.window = sg.get_window('hann', Parm.M);        # Window in Vector Form
Parm.N = 8192;                                      # Analysis DFT Size, 185.76ms
Parm.H = 512;                                       # Hop Size, 11.61ms
Parm.fs = 44100;                                    # Sampling Rate, 44.10K Hz
Parm.t = 42                                         # Dicard Peaks below Mag level 42
# PT algo
Parm.binFreq = Parm.fs/Parm.N                       # Frequency of a Bin
Parm.freqDevSlope = 0.01                            # Slope of the frequency deviation
Parm.freqDevOffset = 30                             # The minimum frequency deviation at 0 Hz
Parm.MagCond = 4                                    # 4 dB
Parm.minPartialLength = 4                           # Min Partial length, 4 peaks, 46.44ms

#########################################################################
## Step 0 - Obtain Audio File Name
WavFileNames = DB.iKalaWavFileNames(WavDirStr);
numMusics = len(WavFileNames);

for t in np.arange(numMusics):
    #########################################################################
    ## Step 1 - Import Audio and Create Power Spectrogram
    tic = time.time()
    # import audio
    x, fs = AD.audioread(WavFileNames[t])
    Voice = Signal()
    Song = Signal()
    Mix = Signal()
    Voice.x = x[:,1]
    Song.x = x[:,0]
    Mix.x = x[:,0]+x[:,1]
    ## For Synthesize, constraint the amplitude to either the original max min, or [-1, 1]
    MinAmp = Mix.x.min(0) 
    if MinAmp<-1: 
        MinAmp = -1
    MaxAmp = Mix.x.max(0)
    if MaxAmp>1:
        MaxAmp = 1
    # Spectrogram Dimension - Parm.numBins:4097 X Parm.numFrames:2584 = 10,586,648
    _, Voice.mX, _, _, _, _ = SM.stft(Voice.x, Parm)
    _, Song.mX, _, _, _, _ = SM.stft(Song.x, Parm)
    _, Mix.mX, Mix.pX, Parm.remain, Parm.numFrames, Parm.numBins = SM.stft(Mix.x, Parm)
    Mix.mXdB = UT.MagTodB(Mix.mX);
    Parm.mindB = min(Mix.mXdB.min(0)) 
    Parm.maxdB = max(Mix.mXdB.max(0))
    toc = time.time() - tic
    if t < 137:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) )
    else:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) )

    #########################################################################
    ## Step 2 - Create Ideal Binary Mask
    tic = time.time()  
    Voice.IBM = Voice.mX > Song.mX
    Song.IBM = Voice.mX <= Song.mX
    Mix.ploc = SM.peakDetection( Mix.mXdB, Parm )
    
    Voice.IBMPeak = Voice.IBM * Mix.ploc;
    Song.IBMPeak = Song.IBM * Mix.ploc;
    toc = time.time() - tic
    print('Create IBM needs %.2f sec' % toc)

    #########################################################################
    ## Step 3 - Create Sinusoidal Partials
    tic = time.time()
    Partials = SM.PT_Algo_FM_C( Mix.mXdB.transpose(), Mix.ploc.transpose(), Voice.IBMPeak.transpose(), Parm )
    toc = time.time() - tic
    print('Create Sinusoidal Partials needs %.2f sec' % toc);

    #########################################################################
    ## Step 4 - Classify Sinusoidal Partials
    tic = time.time()
    Voice.Partials, Song.Partials = PU.ClassifyPartials( Partials )
    toc = time.time() - tic
    print('Classify Sinusoidal Partials need %.2f sec' % toc)

    #########################################################################
    ## Step 5 - iSTFT/AddSynth
    tic = time.time()
    mV = FT.PartialsToBinaryMask( Voice.Partials, Parm ) * Mix.mX;
    mV[mV<np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    mS = FT.PartialsToBinaryMask( Song.Partials, Parm ) * Mix.mX;
    mS[mS<np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    Voice.IBMPeaky = SM.istft(mV, Mix.pX, Parm )
    Voice.IBMPeaky = AD.scaleAudio( Voice.IBMPeaky, MinAmp, MaxAmp )
    Song.IBMPeaky = SM.istft(mS, Mix.pX, Parm )
    Song.IBMPeaky = AD.scaleAudio( Song.IBMPeaky, MinAmp, MaxAmp )
    
    Voice.PMask = FT.PartialsToBinaryMask( Voice.Partials, Parm );
    mVdB = Voice.PMask * Mix.mXdB;
    mVdB = FT.prepareSineSynth( mVdB, Voice.PMask, Parm );
    pV = FT.prepareSineSynth( Mix.pX, Voice.PMask, Parm );
    Song.PMask = FT.PartialsToBinaryMask( Song.Partials, Parm );    
    mSdB = Song.PMask * Mix.mXdB;
    mSdB = FT.prepareSineSynth( mSdB, Song.PMask, Parm );
    pS = FT.prepareSineSynth( Mix.pX, Song.PMask, Parm );
    Voice.IBMPeakSiney = SM.sineSynth( mVdB, pV, Voice.PMask, Parm );
    Voice.IBMPeakSiney = AD.scaleAudio( Voice.IBMPeakSiney, MinAmp, MaxAmp );
    Song.IBMPeakSiney = SM.sineSynth( mSdB, pS, Song.PMask, Parm );
    Song.IBMPeakSiney = AD.scaleAudio( Song.IBMPeakSiney, MinAmp, MaxAmp );
    
    toc = time.time() - tic
    print('Computing iSTFT need %.2f sec' % toc)
    
    #########################################################################
    ## Step 4 - genSound
    tic = time.time()
    if t < 137:
        AD.audiowrite(AudioOutDirStr+'/FM_iSTFT_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Voice_'+WavFileNames[t][-15:], Voice.IBMPeaky, fs );
        AD.audiowrite(AudioOutDirStr+'/FM_iSTFT_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Song_'+WavFileNames[t][-15:], Song.IBMPeaky, fs );
        AD.audiowrite(AudioOutDirStr+'/FM_AS_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Voice_'+WavFileNames[t][-15:], Voice.IBMPeakSiney, fs );
        AD.audiowrite(AudioOutDirStr+'/FM_AS_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Song_'+WavFileNames[t][-15:], Song.IBMPeakSiney, fs );
    else:
        AD.audiowrite(AudioOutDirStr+'/FM_iSTFT_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Voice_'+WavFileNames[t][-16:], Voice.IBMPeaky, fs );
        AD.audiowrite(AudioOutDirStr+'/FM_iSTFT_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Song_'+WavFileNames[t][-16:], Song.IBMPeaky, fs );    
        AD.audiowrite(AudioOutDirStr+'/FM_AS_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Voice_'+WavFileNames[t][-16:], Voice.IBMPeakSiney, fs );
        AD.audiowrite(AudioOutDirStr+'/FM_AS_2048_8192_512_42_001_30_4_4/'+str(t+1)+'_Song_'+WavFileNames[t][-16:], Song.IBMPeakSiney, fs );
    toc = time.time() - tic
    print('genSound needs %.2f sec' % toc)
