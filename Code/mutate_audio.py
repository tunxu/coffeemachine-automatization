import os
import torch
import torchaudio


mutated_audio = []



with open("mutated.txt", "rt") as file:
    for line in file:
        mutated_audio.append(line)


with open('mutated.txt', "w+") as mutated_file:
    print('Successfully opened mutated.txt')
    for root, dir, files in os.walk("data2", topdown=False):
        for name in files:
            if name not in mutated_audio:
                audio_tensor, sample_rate = torchaudio.load(os.path.join(root, name))
                if audio_tensor.shape[1] == 2:
                    mono_audio = audio_tensor[:, :-1]
                    audio_tensor = mono_audio
                    print("Succeeded in turning to mono channel")
                
                pitch_transformer_up = torchaudio.transforms.PitchShift(sample_rate, 2)
                pitch_transformer_down = torchaudio.transforms.PitchShift(sample_rate, -2)

                pitched_audio = pitch_transformer_up(audio_tensor)
                torchaudio.save(os.path.join(root, f"{name[:-4]}_pitch_up.wav"), pitched_audio.detach(), sample_rate=sample_rate)     

                pitched_audio = pitch_transformer_down(audio_tensor)  
                torchaudio.save(os.path.join(root, f"{name[:-3]}_pitch_down.wav"), pitched_audio.detach(), sample_rate=sample_rate) 
                mutated_file.write(f"{name}\n{name[:-4]}_pitch_up.wav\n{name[-4]}_pitch_down.wav")
        

