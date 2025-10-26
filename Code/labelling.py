import os
import json

files = os.listdir(os.path.join("data"))

audio_class_dict = {}

for element in files:
    if element.endswith(".wav"):
        match element:
            case s if s.startswith("fernando"):
                audio_class_dict.update({element : "fernando"})
            case s if s.startswith("gross"):
                audio_class_dict.update({element : "gross"})
            case s if s.startswith("klein"):
                audio_class_dict.update({element : "klein"})
            case s if s.startswith("ok"):
                audio_class_dict.update({element : "ok"})
            case s if s.startswith("kaffee"):
                audio_class_dict.update({element : "kaffee"})

with open ("data/audio_label.json", "w") as file:
    json.dump(audio_class_dict, file)

print(audio_class_dict.keys())