params = {'font.size': 14,
          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 12,
#          'legend.fontweight': 'bold',
         }

# Global parameters
ext='.mp4' # '.avi'
beltSpeed=10
#model name
model = 'DLC_resnet50_V2a_Swimming_TopViewNov11shuffle1_1030000'

# Marker ids used in DeepLabCut
mrkr = ['head','spine 1','spine 2','spine 3','spine 4','base',
        'tail 25','tail 50','tail 75','tail 100']

# Markers used in speed estimation. Excludes paws which are used in coordination
speedMarkers = mrkr[1:3]
bodyLen = 90 #mm
length = 450 # Width of the image in mm
# Time points
time_points = range(49,115,7)
time_points = ['P'+repr(i) for i in time_points ]

# Smoothing window for position estimates
smFactor = 30
# Smoothing window for speed estimates
speedSmFactor = 50
speedThr = 5 # Used to leave out stride and cadence calcuations
# Acceleration smoothing params
tThr = 0.25 # Duration to count a drag/recovery event
accSmFactor = 12
# Location to save speedProfiles
spProfLoc = '../labels'
#spProfLoc = '../speedProfile'
acProfLoc = '../accelProfile'
cdProfLoc = '../coordProfile'

# Interpolation factor
INTERP = 4

# Keys for archive
keys=['speed','lCad','rCad','flCad','frCad','avg','rStLen','lStLen',
      'frStLen','flStLen','phi','R','nSteps','phi_h','R_h','phi_xR','R_xR',
      'phi_xL','R_xL', 'phi_fLhR', 'R_fLhR','phi_fRhL','R_fRhL',
      'movDur',  'rStride','lStride','fRStride','fLStride']
colors=['black','blue','green','grey']
legends=['1st','2nd','3rd','Mean']
locKeys = ['LH_RH','LH_LF','RH_RF','LF_RH','RF_LH']

## For making video overlays
frameRate = 24
