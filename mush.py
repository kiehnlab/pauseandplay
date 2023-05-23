#from __future__ import division
#from __future__ import print_function
#import warnings
#warnings.filterwarnings("ignore")

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
from tools import videoMetadata
from constants import params,smFactor
import matplotlib
matplotlib.rcParams.update(params)
from matplotlib.pyplot import cm
import matplotlib._color_data as mcd
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tools import circular_mean

# Constants
model ='DLC_resnet50_haizeaApr4shuffle1_75000'
## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='LOCATION',help='Path to video files.')
parser.add_argument('--meta', type=str, default='METADATA.xlsx',help='Path to spreadsheet with metadata.')

args = parser.parse_args()
groups = ['ppn','pag']
print("Analyzing videos in location"+args.data)
fig1 = plt.figure(figsize=(16,8))
gs1 = GridSpec(2,4,figure=fig1)

gIdx = -1

for g in groups:
    print('Processing '+g+' ...')
    gIdx += 1
    all_meta = pd.read_excel(sorted(glob.glob(args.data+g+'/*.xlsx'))[0])
    vids = sorted(glob.glob(args.data+g+'/*.avi'))
    os.chdir(args.data+g+'/labels')
    uniq_anim = [vid.split('/')[-1].split('_')[1] for vid in vids]

    data = pd.read_hdf(vids[0].replace('.avi','.h5'))
    pairs = ['LHRH']

    phase = pd.DataFrame(index=np.arange(len(vids)),columns=pairs)
    phase['name'] = list(uniq_anim)
    phase['color'] = None

    uniq_anim = np.array(np.unique(uniq_anim))

    colors = [name for name in mcd.XKCD_COLORS]
    vIdx = -1
    cIdx = 0

    for vid in vids:
        vIdx += 1
        print("Processing "+vid)
        animIdx = vid.split('/')[-1].split('_')[1]
        color = colors[(np.where(uniq_anim == animIdx)[0][0]+10)*2]
        meta = videoMetadata(vid)
        light_on = all_meta[all_meta.Raw_filename == vid.split('/')[-1]].light_on_f.values[0]
        phCor = all_meta[all_meta.Raw_filename == vid.split('/')[-1]].phCor.values[0]
        ipFile = vid.replace('.avi','.h5')
        data = pd.read_hdf(ipFile)
        phase['color'][vIdx] = color
        for p in pairs:
            offset = 0
            speed0 = data[p+'_zero_speed'].values[0]
            speed_nz = data[p+'_nz_speed'].values[0]
            time0 = speed0/meta['fps']-0.3
            ax = plt.subplot(gs1[gIdx,:])
            locData = data[p].values[:500]
            lMax = locData.max()
            lMin = locData.min()
            locData = (180*(2*(locData-lMin)/(lMax-lMin)-1))
            if speed0 > 150:
                speed0 = speed0-60


            if phCor > 0:
                idx = np.where(locData[:speed0] > (locData[speed0]+phCor))[0]
            else:
                idx = np.where(locData[:speed0] < (locData[speed0]+phCor))[0]
            if len(idx) > 0:
                idx = idx[-1]
                newLocData = np.zeros(len(locData))
                offset = int(speed0-idx)

                print(speed0,phCor,offset)
                locData = data[p].values[:(500)]
                if 'Chx10-R26ChR_273_vlPAG_200820_P018_10ms_40Hz_5' in vid:
                    locData[speed0:] = locData[speed0]-locData[-1]

                newLocData[offset:speed0] = locData[:idx]
                newLocData[speed0:] = locData[speed0:] + phCor


                print(locData[speed0],phCor,newLocData[speed0])    
                locData = newLocData.copy()
            ### Renormalize
            SM = 2
            locData[speed0+SM//2:(-SM+1)] = np.convolve(locData[speed0+SM//2:],np.ones(SM,)/SM,'valid')

            locData = (180*(2*(locData-locData.min())/(locData.max()-locData.min())-1))
            time = np.linspace(-0.3,1.3,len(locData))
            phase.loc[idx,p] = locData[250]
            plt.plot(time[offset:],locData[offset:],color='tab:grey',alpha=0.5)

            plt.title(p)
            if vIdx == (len(vids)-1):
                plt.plot(np.ones(10)*0,np.linspace(-180,180,10),'--',
                        color='tab:blue',alpha=0.9,linewidth=4)
                plt.plot(np.ones(10),np.linspace(-180,180,10),'--',
                        color='tab:blue',alpha=0.9,linewidth=4)
            plt.xlim([-0.3,1.3])
            plt.yticks(np.linspace(-180,180,5),[-180,-90,0,90,180])
            plt.title(g)
    band = 0*np.ones(len(time))

plt.tight_layout()
plt.savefig(args.data+'mustache.pdf')



