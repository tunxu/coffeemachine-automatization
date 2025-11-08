import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


#audio, sample_rate = torchaudio.load('data2/fernando/fernando.wav') #Open Audio File

#def set_set_to_16000(tensor):
#    if tensor.size()[0] != 16000:
#        tensor = tensor[:160000, ...]
#
#
#
#
for root, dir, files in os.walk("data2", topdown=False):
    for name in files:
        audio_tensor, sample_rate = torchaudio.load(os.path.join(root, name))
        if audio_tensor.size()[1] > 20000:
            print(name[:-4])



## Resampling
#
#resample_rate = 16000
#
#for root, dir, files in os.walk('data2', topdown=False):
#    for name in files:
#        waveform, sample_rate = torchaudio.load(os.path.join(root, name))





#pitch_transformer = torchaudio.transforms.PitchShift(sample_rate, -2)
#pitched_audio = pitch_transformer(audio)
#
#torchaudio.save("something.wav", pitched_audio.detach(), sample_rate=sample_rate)

"""
        waveform, sample_rate = torchaudio.load(os.path.join(root, name))
        resampled_waveform = F.resample(
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
"""