import os
os.environ['DLClight'] = 'True'
import glob
import deeplabcut

LOC='recordings/'
CONFIG=LOC+'/config.yaml'
### Train DLC model with parameters in config.yaml

deeplabcut.train_network(config=CONFIG)

folders = sorted(glob.glob(location))
BEHAVIOUR='/rearing'
for f in folders:
    print('Processing '+f)
    deeplabcut.analyze_videos(config=CONFIG,videos=[f+BEHAVIOUR])
    deeplabcut.filterpredictions(config=CONFIG,video=[f+BEHAVIOUR],filtertype='arima')
    deeplabcut.create_labeled_video(config=CONFIG,videos=[f+BEHAVIOUR],filtered=True)
