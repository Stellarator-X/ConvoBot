import argparse

parser = argparse.ArgumentParser(description="Tests the Data Generator and Data Augmentation", 
                                 epilog ="Displays the time series/spectrogram of the augmented sample/Instantiates the Data Generator")

group1 = parser.add_argument_group('Augmentation Tests')
group1.add_argument('--audio_file_path', default = None, help = 'Path to the audio file')
group1.add_argument("-a","--augmentation", type=str, choices=['spec', 'timeser'], default='spec',
                    help = 'Type of augmentation : spec => on spectrogram, timeser => on time series')

group2 = parser.add_argument_group('Data Generation Tests')
group2.add_argument('--dataset_path', default=None, help = 'Path to the dataset')
group2.add_argument('-d', "--data_gen", action='store_true', help = "Creates a Data Generator with the given dataset")

args = parser.parse_args()

from ds_utils.augmentation import *
from ds_utils.data_manip import *
import os 
os.system("clear")
if args.audio_file_path:
    print("Generating Augmentations:")
    spect = to_spectrogram(args.audio_file_path)
    print("Spectrogram of shape:", spect.shape)
    aug.show_spectrogram(spect)
    augspect, _ = specAugment([spect], ["No"])
    augspect = augspect[1]
    print("Augmented spectrogram of shape:", augspect.shape)
    aug.show_spectrogram(augspect.numpy())

if args.dataset_path:
    data = get_data(path =args.dataset_path +'/', verbose = True)
    print("First sample : ",data[0])
    print("Initialised Data Generator : ", end = "")
    a = SR_DataGenerator(data[:,0], data[:,1])
    print(a)