import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.signal import find_peaks

def peaker(data):
    peaks,_ = find_peaks(data,prominence=0.5)
    peakDist = np.mean(np.diff(peaks)) /2
#    peaks, _ = find_peaks(data,distance=peakDist)
    peaks = [0] + list(peaks) + [len(data)-1]
    return peaks

def offset_peaks(data):
    peaks = peaker(data)
    for k in range(1,len(peaks)):
        lIdx = peaks[k-1]
        uIdx = peaks[k]
        lin_curve = np.linspace(data[lIdx],data[uIdx], uIdx-lIdx)
        data[lIdx:uIdx] -= lin_curve
    return data

def videoMetadata(vid):
    meta = {}
    cap = cv2.VideoCapture(vid)
    meta['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    meta['nFrame'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta['dur'] = round(meta['nFrame']/meta['fps'],2)
    meta['imW'] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    meta['imH'] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    meta['xPixW'] = length/meta['imW']

    return meta

def trimVid(vid):
    """
    Trim videos to include frames that have the 
    animal visible
    """
    cap = cv2.VideoCapture(vid)
    fps = np.int(cap.get(cv2.CAP_PROP_FPS))
    N = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = np.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = np.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.zeros((N))

    for k in range(N):
        flag, frame = cap.read()
        frames[k] = frame.mean()
        
    pdb.set_trace()
    out = cv2.VideoWriter(vid.replace('.avi','_cropped.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (W,H))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def circular_mean(phi,r=np.ones(1),ignore_zeros=True):
    if ignore_zeros:
        phi = phi[phi != 0]
        if len(r) != 1:
            r = r[phi != 0]
    X = (r*np.cos(phi)).mean()
    Y = (r*np.sin(phi)).mean()
    meanR = np.sqrt(X**2+Y**2)
    meanPhi = np.arctan2(Y,X)
    return meanPhi, meanR

def iqrMean(data):
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = upper_quartile-lower_quartile
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = data[np.where((data >= quartileSet[0]) & (data <= quartileSet[1]))]
    
    return result.mean()
