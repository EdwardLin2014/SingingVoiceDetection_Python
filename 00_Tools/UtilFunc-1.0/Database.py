import numpy as np
from os import walk

def iKalaWavFileNames(DatabaseDirStr):
    '''
    Output 
        FileDirs = cell(252,1);
        137 Verse = cell(1:137,1);
        115 Chorus = cell(138:252,1);
    '''
    ## Function Body
    iKala = []
    for (dirpath, dirnames, filenames) in walk(DatabaseDirStr):
        iKala.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(iKala):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    
    FilesDirs = ["" for x in range(numFiles)]
    l = -1;
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        wavname = iKala[i]
        if wavname[-10:] == '_verse.wav':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname
    for i in np.arange(startIdx+1,numFiles+startIdx):
        wavname = iKala[i]
        if wavname[-11:] == '_chorus.wav':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname

    return FilesDirs

def iKalaPitchLabelFileNames(DatabaseDirStr):
    '''
    Output 
        FileDirs = cell(252,1);
        137 Verse = cell(1:137,1);
        115 Chorus = cell(138:252,1);
    '''
    ## Function Body
    iKala = []
    for (dirpath, dirnames, filenames) in walk(DatabaseDirStr):
        iKala.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(iKala):
        if len(filename) > 3:
            if filename[-3:] == '.pv':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    
    FilesDirs = ["" for x in range(numFiles)]
    l = -1;
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        wavname = iKala[i]
        if wavname[-9:] == '_verse.pv':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname
    for i in np.arange(startIdx+1,numFiles+startIdx):
        wavname = iKala[i]
        if wavname[-10:] == '_chorus.pv':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname

    return FilesDirs
    
def iKalaPitchMask( PitchFileNames,numMusics ):
    PitchMask = np.zeros((numMusics,937),float);
    
    for n in np.arange(numMusics):
        with open(PitchFileNames[n]) as f:
            i = 0
            for line in f:
                PitchMask[n,i] = line
                i += 1
    PitchMask[PitchMask>1] = 1;
    return PitchMask
