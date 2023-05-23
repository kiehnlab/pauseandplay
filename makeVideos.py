import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import time
import os
from tools import videoMetadata
from constants import *
import pdb 
from scipy.stats import circmean
import matplotlib
from matplotlib import gridspec
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb

params = {'font.size': 24,
          'font.weight': 'normal',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'normal',
          'axes.titleweight':'normal',
          'legend.fontsize': 14,
         }
matplotlib.rcParams.update(params)

newcolors_rear = np.array([
            [233, 99, 170], #light pink
            [45, 114, 143],#royalblue_light
            [245, 176, 65], #yellow
            ])

newcolors_groom = np.array([
            [233, 99, 170], #light pink
            [53, 92, 125], #royalblue
            [45, 114, 143],#royalblue_light
            [245, 176, 65], #yellow
            [100, 17, 89], #magenta
            [185, 27, 111],#pink
            ])

## MAIN PROGRAM STARTS HERE ## 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/home/raghav/dsl/projects/kiehn_lab/haizea/cylinder/video_vis/final_set/Trail10-s517/', help='Path to labeled videos')
parser.add_argument('--colors', type=str, default='',
            help='Color schemes: sod,cno,both')
parser.add_argument('--start', type=float, default=0,
            help='Video starting time')
parser.add_argument('--end', type=float, default=-1,
            help='Video ending time')
parser.add_argument('--behaviour', type=str, default='Rearing ', help='Rearing or Grooming')

args = parser.parse_args()
print("Analyzing coordination from "+args.data)
if 'Rearing' in args.data:
    mrkr = ['snout','LF','RF']
    mrkrExp = ['snout', 'left paw', 'right paw']
    figOffset = 1
    newcolors = newcolors_rear/ 255.0
else:
    mrkr = ['snout','lf','rf','l-ear','r-ear']
    mrkrExp = ['snout','left paw', 'right paw','left ear', 'right ear']
    newcolors = newcolors_groom/255.0
    figOffset = 0

os.chdir(args.data)
files = sorted(glob.glob('*labeled.mp4'))
for ipFile in files:
    meta = videoMetadata(ipFile)
    sFrame = 0
    eFrame = meta['nFrame']
    dur = meta['dur']
    if not os.path.exists('overlays'):
        os.mkdir('overlays')

    data_file = sorted(glob.glob('*_filtered.h5'))[0] 
    #pdb.set_trace()
    data = pd.read_hdf(data_file)
    model = data.keys()[0][0]
    T = len(data)
    light_on = int(meta['fps']*5)
    light_off = light_on + int(meta['fps']*3)
    T_s = light_on - int(meta['fps']*3)
    T_e = light_on + int(meta['fps']*6)

    ### Initialize output frame
    fig = plt.figure(figsize=(35,20))
    gs = gridspec.GridSpec(6, 2)
    frames = sorted(glob.glob(args.data+'frames/*.jpg'))
    frame = plt.imread(frames[0])
    assert len(frames) != 0, "No video frames found. Use ffmpeg to extract frames!"
    assert len(frames) == T, "Tracked data does not match number of frames!"
    imH, imW = frame.shape[0], frame.shape[1]
    colors = newcolors #['tab:red','tab:blue','tab:green']
    frames = frames[T_s:T_e]
    data = data[T_s:T_e]
    T = len(data)
    light_on = light_on - int(meta['fps']*2)
    light_off = light_off  - int(meta['fps']*2)

    ### DLC Video frame in output frame
    for i in range(T):
        plt.clf()
                ### Plot video frame
        ax0 = plt.subplot(gs[:,0]) #plt.subplot(4,1,1)
        frame = plt.imread(frames[i])
        plt.imshow(frame)

        if (i >= light_on) and (i < light_off):
            xaxis = np.arange(imW-100,imW)
            band = np.ones(len(xaxis))*(imH-51)
            plt.plot(xaxis,band,linewidth=0.,color=[0,0,1])
            plt.fill_between(xaxis,band-50,band+50,color=[0,0,1])
        plt.xticks([])
        plt.yticks([])
        plt.title(args.data.split('/')[-2].replace('_',' '), fontdict={'size':32},pad=50)

        xAxis = np.linspace(-3,6,T)
        ### Speed profile in output frame

        for mIdx in range(len(mrkr)):
            m = mrkr[mIdx]         
            ax1 = plt.subplot(gs[figOffset+mIdx,1])#    plt.subplot(4,1,2)
            if mIdx == 0:
                plt.title('Y-position profiles', fontdict={'size':32},pad=30)# for '+videoName)
            plt.plot(xAxis,meta['imH']-data[model][m]['y'],\
                    color='tab:grey',alpha=0.3,linewidth=3)
            plt.plot(xAxis[:i],meta['imH']-data[model][m]['y'][:i],color=colors[mIdx],
                      linewidth=5,label=mrkrExp[mIdx])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper right')
            plt.xlim([-3,7.2])
        
            ax2 = plt.subplot(gs[-1-(figOffset),1])#    plt.subplot(4,1,2)

            plt.plot(xAxis,meta['imH']-data[model][m]['y'],\
                    color='tab:grey',alpha=0.3,linewidth=3)
            plt.plot(xAxis[:i],meta['imH']-data[model][m]['y'][:i],color=colors[mIdx],
                      linewidth=5,label=mrkrExp[mIdx])
            plt.legend(loc='upper right')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            plt.xticks([-3,0,3,6],[-3,0,3,6])
            plt.xlim([-3,7.2])

        plt.xlabel('Time from light onset (s)',fontdict={'size':24})
        plt.tight_layout()

        plt.yticks([])
        plt.savefig('overlays/fig'+'{0:05d}'.format(i)+'.jpg');
