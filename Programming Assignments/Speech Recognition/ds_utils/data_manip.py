import os
import numpy as np 
import tensorflow as tf 
import augmentation as aug

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

"""
Data Generator
"""

class SR_DataGenerator(tf.keras.utils.Sequence):
    """
        Keras based DataGenerator class for speech recognition datasets
    """
     # TODO @stellarator-x Arrange in increasing order for first epoch

    def __init__(self, spectIDs, labels, batch_size = 32, dims = (1025, 1050), augmentation_ratio = 2):
        """
        augmentation_ratio : #samples(augdat)/#samples(dat) : choice[1, 2, 3]
        """
        self.spectIDs = spectIDs
        self.labels = labels
        self.dims = dims
        self.batch_size = batch_size
        self.augmentation_ratio = augmentation_ratio


    def __len__(self):
        return int(np.floor(len(self.spectIDs) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.spectIDs))
        if self.shuffle is True:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, spects_temp):
        X = np.empty((self.batch_size, *self.dims))
        y = np.empty((self.batch_size), dtype = str)

        for i, ex in enumerate(spects_temp):
            X[i,] = None # Load from <path_to_id>
            y[i] = self.labels[ID]

        # Augment X
        if self.augmentation_ratio is not 1:
            X, y = aug.specAugment(X, add_random = (self.augmentation_ratio==3))

        return X, y

    def __getitem__(self, index):
        # Generates on batch of data
        indices = self.indices[index*self.batch_size/self.augmentation_ratio : (index+1)*self.batch_size/self.augmentation_ratio]

        spects_temp = [self.spects[k] for k in indices]

        X, y = self.__data_generation(spects_temp)

        return X, y
    

