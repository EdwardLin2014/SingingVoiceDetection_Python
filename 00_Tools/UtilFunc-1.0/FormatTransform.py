import numpy as np

def prepareSineSynth( X, ploc, Parm ):
    '''
    %%
    %   X: numBins*numFrames matrix, e.g. mX, pX
    %   ploc: peak location
    %   Parm: system configuration
    %   retrun cX: cell(1,numFrames)
    '''
    numFrames = Parm.numFrames
    cX = [[]] * numFrames

    for n in np.arange(numFrames):
        cX[n] = X[(ploc[:,n]==1),n]

    return cX



