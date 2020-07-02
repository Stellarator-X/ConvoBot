import argparse

# Adding arguments
parser = argparse.ArgumentParser()
parser.add_argument()

# Dependencies
from ds_utils.augmentation import *
from ds_utils.data_manip import *
from ds_utils.model import *
import matplotlib.pyplot as plt
import os

# Defining checkpoint variables
import os
checkpoint_path = "bin/Training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period = 5)


# History-plotting function
def plothist(history):
    loss_hist = history.history['loss']
    acc_hist = history.history['accuracy']
    f, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(acc_hist, label = 'accuracy')
    plt2.plot(loss_hist, label="loss/max_loss")
    plt1.xlabel("Epochs")
    plt1.ylabel("Loss")
    plt1.xlabel("Epochs")
    plt1.ylabel("Acc")
    plt.legend()
    plt.show()

# Dataset
datset = get_data(verbose=True)
files = datset[:, 0]
file_to_label = { datset[i, 0]:datset[i,1] for i in range(datset.shape[0])}

# Model
ds_model = DSModel(num_conv = 2, num_rnn = 3)
model = ds_model.build()

try:
    ds_model.restore(checkpoint_path)
except:
    print("No previous checkpoints")

speechgen = SpeechDataGenerator(files, file_to_label)

hist = model.fit(speechgen, epochs = 20)

plothist(hist)