import numpy as np
from scipy.linalg import toeplitz
import scipy.signal as sg

"""
    Find 2^n that is equal to or greater than.
    This is internal function used by fft(), because the FFT routine
    requires that the data size be a power of 2.
"""
def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    
    return n

def bss_eval_1source(se,s):
    ### Performance criteria ###
    s_true,e_spat,e_interf,e_artif = bss_decomp_mtifilt(se,s,512);
    SDR, SIR, SAR = bss_source_crit(s_true,e_spat,e_interf,e_artif);
    
    return SDR, SIR, SAR

def bss_decomp_mtifilt(se,s,flen):
    ### Decomposition ###
    # True source image
    s_true = np.append(s,np.zeros((flen-1,1)))
    # Spatial (or filtering) distortion
    e_spat = project(se,s,flen) - s_true;
    # Interference is always zeros
    e_interf = np.zeros((e_spat.shape));
    # Artifacts
    e_artif = np.append(se,np.zeros((1,flen-1))) - s_true - e_spat;
    
    return s_true,e_spat,e_interf,e_artif

def project(se,s,flen):
    # SPROJ Least-squares projection of each channel of se on the subspace
    # spanned by delayed versions of the channels of s, with delays between 0
    # and flen-1
    nsampl = s.size
    
    ### Computing coefficients of least squares problem via FFT ###
    # Zero padding and FFT of input data
    s = np.append(s,np.zeros((flen-1,1)))
    se = np.append(se,np.zeros((flen-1,1)))
    fftlen = nextpow2(nsampl+flen-1);
    sf = np.fft.fft(s,fftlen);
    sef = np.fft.fft(se,fftlen);
    # Inner products between delayed versions of s
    ssf = np.multiply(sf,sf.conjugate()).real
    ssf = (np.fft.ifft(ssf)).real; 
    ssfA = ssf[0]
    for idx in range(fftlen-1,fftlen-flen,-1):
        ssfA = np.append(ssfA,ssf[idx])
    ssfB = ssf[0:flen]
    ss = toeplitz(ssfA,ssfB);
    G = np.transpose(ss);
    
    # Inner products between se and delayed versions of s
    ssef = np.multiply(sf,sef.conjugate()).real
    ssef = (np.fft.ifft(ssef)).real;    
    ssefA = ssef[0]
    for idx in range(fftlen-1,fftlen-flen,-1):
        ssefA = np.append(ssefA,ssef[idx])   
    D = np.transpose(ssefA);
    
    ### Computing projection ###
    # Distortion filters
    C = np.transpose((np.linalg.inv(G)).dot(D))
    # Filtering
    sproj = sg.filtfilt(C,C,s);
    
    return sproj

def bss_source_crit(s_true,e_spat,e_interf,e_artif):
    ### Energy ratios ###
    s_filt = s_true + e_spat;
    numerator = ((s_filt**2).sum(axis=0)).sum(axis=0)
    # SDR
    SDR = 10*np.log10(numerator/(((e_interf+e_artif)**2).sum(axis=0)).sum(axis=0));
    # SIR
    SIR = 10*np.log10(numerator/((e_interf**2).sum(axis=0)).sum(axis=0));
    # SAR
    SAR = 10*np.log10( ((((s_filt+e_interf)**2).sum(axis=0)).sum(axis=0))/(((e_artif**2).sum(axis=0)).sum(axis=0)) );
    
    return SDR,SIR,SAR