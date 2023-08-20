# Colab functions

import os
import json

def colab_clean_up(input_file_name):
    for item in [input_file_name, "mapthis.json", "audio.mp3", "timing.osu", "rhythm_data.npz", "mapthis.npz"]:
        try:
            os.remove(item)
        except:
            pass
    print("intermediate files cleaned up!")

def load_pretrained_model(model_name):
    model_data = json.load(open("presets.json"))

    if model_name not in model_data:
        return model_data["default"]
    return model_data[model_name]