import os
import numpy as np 
import tensorflow as tf 

def find_files(root_search_path, files_extension):
    files_list = []
    for root, _, files in os.walk(root_search_path):
        files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
    return files_list

def clean_label(_str):
        _str = _str.strip()
        _str = _str.lower()
        _str = _str.replace(".", "")
        _str = _str.replace(",", "")
        _str = _str.replace("?", "")
        _str = _str.replace("!", "")
        _str = _str.replace(":", "")
        _str = _str.replace("-", " ")
        _str = _str.replace("_", " ")
        _str = _str.replace("  ", " ")
        return _str

def get_data(path = 'LibriSpeech/'):
    text_files = find_files(path, ".txt")
    data = []
    for text_file in text_files:
        directory = os.path.dirname(text_file)
        with open(text_file, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                head = line.split(' ')[0]
                if len(head) < 5:
                    # Not a line with a file description
                    break
                audio_file = directory + "/" + head + ".flac"
                if os.path.exists(audio_file):
                    data.append([audio_file, clean_label(line.replace(head, "")), None])
    
    data = np.array(data)
    data = data[:, :-1] # The last index is NoneType
    print(f"Loaded dataset with shape {data.shape}")
    return data

class SR_DataGenerator(tf.keras.utils.Sequence):
    """
        Keras based DataGenerator class for speech recognition datasets
    """

    def __init__(self, bunch_of_stuff):
        #init bunch
        pass

    def __len__(self):
        # batches per epoch
        pass

    def __data_generation(self, list_ids):
        # 
        pass