import numpy as np
import pypianoroll as pl
import os
import os.path
import scipy.io.wavfile as wavf
from matplotlib import pyplot as plt

# for root, dirs, files in os.walk("../../", topdown=False):
#     for name in files:
#         data = pl.load(os.path.join(root.replace("\\","/"), name))
#         if(len(data.tracks) == 1):
#             midi = data.to_pretty_midi()
#             audio = midi.synthesize()
#             wavf.write(data.name+".wav",int(len(audio)/(data.get_active_length()/data.beat_resolution)),audio)
#
#             # fig, ax = data.plot()
#             # plt.show()

# pianoroll = np.zeros((96, 128))
# C_maj = [60, 61, 62, 63, 64, 65, 66,67,68,69,70,71,72]
# pianoroll[0:95, C_maj] = 100
#
# # Create a `pypianoroll.Track` instance
# track = pl.Track(pianoroll=pianoroll, program=0, is_drum=False,
#               name='my awesome piano')
# # Plot the pianoroll
# fig, ax = track.plot()
# plt.show()
count = 0
lets_break = False
progress = 0

for dirpath, dirnames, filenames in os.walk("../../lpd/lpd_cleansed/"):
    for filename in [f for f in filenames if f.endswith(".npz")]:
        data = pl.load(os.path.join(dirpath.replace("\\","/"), filename))
        if len(data.tracks) ==1:
            if data.tracks[0].program >= 0 and data.tracks[0].program <= 8:
                count += 1
        progress += 1
        print(str(progress)+"/"+str(21425)+" ("+str(round(((100*progress)/21425),2))+"%)",end="\n")


print(count) # 443

