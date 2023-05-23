from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import pickle
import os
import shutil
import glob
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tools import videoMetadata, offset_peaks, peaker
from constants import params,smFactor
import matplotlib
matplotlib.rcParams.update(params)
from scipy.signal import find_peaks

def getBodyAxis(data): 
    ### Get body axis angle
    dataX = []
    dataY = []
    for m in ['base','head','RF','LF']:
        xData = data[model][m]['x'] 
        yData = data[model][m]['y'] 

        xData = np.convolve(xData,np.ones(smFactor,))/smFactor
        xData = xData[smFactor:-smFactor]
        yData = np.convolve(yData,np.ones(smFactor,))/smFactor
        yData = yData[smFactor:-smFactor]

        L = len(yData)
        L = len(yData)
        dataX.append(xData)
        dataY.append(yData)
    dataX = np.array(dataX)[np.newaxis,:,:]
    dataY = np.array(dataY)[np.newaxis,:,:]
    
    mrkPos = np.concatenate((dataX,dataY))

    ### Obtain head marker using RF,LF,Head
    mrkPos = np.concatenate((mrkPos[:,[0],:],mrkPos[:,1:,:].mean(1,keepdims=True)),axis=1)
    theta =  np.arccos(((mrkPos[0]*mrkPos[1]).sum(0))/(np.linalg.norm(mrkPos[0],axis=0)*np.linalg.norm(mrkPos[1],axis=0)))
    theta[np.isnan(theta)] = 0

    return theta
## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='LOC',help='Path to video files.')
parser.add_argument('--meta', type=str, default='METADATA.xlsx',help='Path to spreadsheet with metadata.')


args = parser.parse_args()
all_meta = pd.read_excel(args.meta)
print("Analyzing videos in location"+args.data)
vids = sorted(glob.glob(args.data+'*.avi'))
os.chdir(args.data+'labels')
limbs = ['LF','LH','RF','RH']

pairs = [list(i) for i in itertools.combinations(limbs,2)]
pairs = [pairs[1],pairs[-2]]
phaseData = pd.DataFrame(index=range(len(vids)),columns=['name',
    'LFRF_pre','LFRF_post','LFRF_phase_diff','LFRF_coherence','LFRF_slope_pre','LFRF_slope_post','LFRF_slope_ratio','LFRF_extreme',
    'LHRH_pre','LHRH_post','LHRH_phase_diff','LHRH_coherence','LHRH_slope_pre','LHRH_slope_post','LHRH_slope_ratio','LHRH_extreme'
    ])
pairNames = [p[0]+p[1] for p in pairs]
cols = [p+'_speed' for p in pairNames] + pairNames
for vIdx in np.arange(len(vids)):
    #plt.clf()
    vid = vids[vIdx]
    phaseData.name[vIdx] = vid.split('/')[-1].split('.')[0]
    print("Processing "+vid)
    meta = videoMetadata(vid)
    light_on = all_meta[all_meta.Raw_filename == vid.split('/')[-1]].light_on_f.values[0]
    ipFile = sorted(glob.glob(vid.split('/')[-1].split('.avi')[0]+'*.h5'))[0]
    data = pd.read_hdf(ipFile)
    phase = all_meta[all_meta.Raw_filename == vid.split('/')[-1]].angle.values[0] 
    model = data.keys()[0][0]
    coord_data = {}
    speed_data = {}
    df = pd.DataFrame(columns=cols,index=np.arange(data.shape[0])) #.from_dict({pair_data[0]})

    theta = getBodyAxis(data)
    for l in limbs:
        lData = data[model][l]['x'].values
        lData = lData[(light_on - meta['fps']//3-smFactor+1):]
        lData = np.convolve(lData, np.ones((smFactor,))/smFactor, mode='valid')
        speed_data[l] = (np.diff(lData))
        lMax = lData.max()
        lMin = lData.min()
        coord_data[l] = lData
    time = np.linspace(-0.3,3.0,len(lData))
    plt.figure(figsize=(16,9))
    pair_data = {}
    pIdx = 1

    for p in pairs:
        ### Pairwise limb speed computation
        pair = p[0]+p[1]
        sData = np.array((speed_data[p[0]],speed_data[p[1]]))
        sData = np.abs(sData).mean(0)
        speedIdx = np.where(sData < 0.15)[0]
        timeOffset = 20

        speed0 = speedIdx[speedIdx > 100][0]
        tmpIdx = np.where(sData[(speed0+200):] > 0.3)[0]
        if len(tmpIdx) == 0:
            speed_nz = speed0
            speed0 = 1
        else:
            speed_nz = np.where(sData[(speed0+200):] > 0.3)[0][0] + (speed0+200) #+ idxOffset
        time0 = time[speed0]
        time_nz = time[speed_nz]
        pData = coord_data[p[0]] - coord_data[p[1]]

        plt.subplot(len(pairs)*2,1,2*pIdx-1)    
        outlier_speed = np.where(sData[400:] > 10)[0]

        idx = 250
        sIdx = 500
        oIdx = 0

        # Filter any sudden increases.
        if idx < speed0:
            idx = speed0+1
        pData[speed0:idx] = np.linspace(pData[speed0],pData[idx],(idx-speed0))
        tmpIdx = pData[:sIdx][pData[:sIdx] >= 0]

        ### If starts with missing marker exculde those points
        if len(tmpIdx) == 0:
            oIdx = np.where(pData > 0)[0][0] + 50
        posMax = pData[:(sIdx+oIdx)][pData[:(sIdx+oIdx)] >= 0].max()
        tmpIdx = pData[:sIdx][pData[:sIdx] < 0]
        if len(tmpIdx) == 0:
            negMin = np.abs(pData[:sIdx+oIdx].min())
        else:
            negMin = np.abs(pData[:(sIdx+oIdx)][pData[:(sIdx+oIdx)] < 0].min())
        pData[pData >0] = pData[pData >0]/posMax
        pData[pData <0] = pData[pData <0]/negMin
        pData[pData > 1] = 1
        pData[pData < -1] = -1
        pData = 180*(pData+1)

        pData = pData - 180
        peaks = find_peaks(pData)[0]
        peaks_pos = list(np.setdiff1d(peaks,peaks[((peaks > (speed0)) \
                & (peaks < (speed_nz)))| (peaks > (speed_nz+idx))]))

        peaks = find_peaks(-pData)[0]
        peaks_neg = list(np.setdiff1d(peaks,peaks[((peaks > (speed0)) \
                & (peaks < (speed_nz)))| (peaks > (speed_nz+idx))]))

        peaks = peaks_pos+peaks_neg
        peaks = np.array(sorted(peaks))
        phase_nzIdx = peaks[peaks > speed_nz][0]
        phase0Idx = peaks[peaks <= speed0][-1]
        phase0 = time[phase0Idx]
        phase_nz = time[phase_nzIdx]
        pair_data[p[0]+p[1]] = pData
        plt.plot(time,pData)
        plt.plot(time[peaks],pData[peaks],'o')
        plt.ylim([-250,250])
        plt.yticks(np.linspace(-180,180,5),[-180,-90,0,90,180])
        plt.plot(np.ones(10)*time0,np.linspace(-180,180,10),
        'tab:grey','--')
        plt.plot(np.ones(10)*time_nz,np.linspace(-180,180,10),
        'tab:grey','--')
        tIdx = np.argmin(np.abs(time-phase0))
        phaseData[pair+'_pre'][vIdx] = pData[speed0]

        extreme = False
        extremeVal = 1

        ## Compute coherence

        if np.abs(speed0-phase0Idx) < timeOffset:
            tmpIdx = peaks[peaks < (speed0-timeOffset)]
            if len(tmpIdx) == 0:
                phase0Idx = peaks[0]
            else:
                phase0Idx = peaks[peaks < (speed0-timeOffset)][-1]
            phase0 = time[phase0Idx]

        if np.abs(speed_nz-phase_nzIdx) < timeOffset:
            phase_nzIdx = peaks[peaks > (speed_nz+timeOffset)][0]
            phase_nz = time[phase_nzIdx]


        
        slope0 = pData[speed0] - pData[phase0Idx]
        slope_nz = pData[phase_nzIdx] - pData[speed_nz]

        pDiff0, pDiff_nz = np.ceil(np.abs(pData[speed0] - pData[phase0Idx])), np.ceil(np.abs(pData[phase_nzIdx]-pData[speed_nz]))
        ph0Thresh = 15
        phNzThresh = 305
        if (pDiff0 > phNzThresh) | (pDiff_nz > phNzThresh) | (pDiff0 < ph0Thresh) | (pDiff_nz < ph0Thresh):
            extremeVal = -1
            extreme = True

        string = 'Phase diff:%.1f, %.1f'%(pDiff0,pDiff_nz)
        string = repr(string)
        phaseData[pair+'_slope_pre'][vIdx] = slope0
        phaseData[pair+'_slope_post'][vIdx] = slope_nz

        phChange = np.sign(slope0/slope_nz)        

        coherence = (1+extremeVal*phChange)/2

        print(string, extreme,coherence)
        phaseData[pair+'_coherence'][vIdx] = coherence
        phaseData[pair+'_extreme'][vIdx] = extreme

        mrkr = '4'
        plt.annotate("",xytext=[time[phase0Idx],pData[phase0Idx]],xy=[time0,pData[speed0]],
                arrowprops=dict(color='tab:red',arrowstyle="->",lw=2))


        phaseData[pair+'_post'][vIdx] = pData[speed_nz]
        plt.annotate("",xytext=[time_nz,pData[speed_nz]],xy=[time[phase_nzIdx],pData[phase_nzIdx]],
                arrowprops=dict(color='tab:red',arrowstyle="->",lw=2))

        df[pair+'_speed'][:len(sData)] = sData
        df[pair+'_zero_speed'] = speed0
        df[pair+'_nz_speed'] = speed_nz
        df[pair][:len(pData)] = pData 
        plt.plot(np.ones(10)*phase0,np.linspace(-180,180,10),
        'tab:green','--')
        plt.plot(np.ones(10)*phase_nz,np.linspace(-180,180,10),
        'tab:green','-x')
        string1 = p[0]+p[1]+' Phase change: %.2f, Extreme=%r, Coherence=%d'%(phChange,extreme,coherence)
#        pdb.set_trace()
        plt.title(string1+'\n'+string)
        # Plot pairwise speed
        plt.subplot(len(pairs)*2,1,2*pIdx)
        plt.plot(time[1:],sData)
        plt.plot(np.ones(10)*time0,np.linspace(sData.min(),sData.max(),10),
        'tab:grey','--')
        plt.plot(np.ones(10)*time_nz,np.linspace(sData.min(),sData.max(),10),
        'tab:grey','--')


        pIdx += 1
    plt.title('Instantaneous speed')

    df.to_hdf(vid.replace('.avi','.h5'), key='df', mode='w')

    plt.tight_layout()
    plt.savefig(vid.replace('.avi','.pdf'))
phaseData['LFRF_phase_diff'] = phaseData['LFRF_post'] - phaseData['LFRF_pre']
phaseData['LHRH_phase_diff'] = phaseData['LHRH_post'] - phaseData['LHRH_pre']
phaseData['LFRF_slope_ratio'] = phaseData['LFRF_slope_pre']/phaseData['LFRF_slope_post']
phaseData['LHRH_slope_ratio'] = phaseData['LHRH_slope_pre']/phaseData['LHRH_slope_post']

phaseData.to_csv(args.data+'phaseChange.csv',index=False)
