import os
import tensorflow_io as tfio
#import numpy as np

for entry in os.scandir("data/me"):
    audio = tfio.audio.AudioIOTensor(os.path.join("data/me", entry.name))
    #turn into mono channel for every file
    if audio.shape[1] == 2:
        mono_audio_tensor = audio.to_tensor()[:, :-1]
    else:
        mono_audio_tensor = audio.to_tensor()

    min_value = min(mono_audio_tensor)
    print(min_value)