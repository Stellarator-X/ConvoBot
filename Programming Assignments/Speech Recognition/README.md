# Speech Recognition : Deep Speech 2

A tensorflow implementation of the Deep Speech 2 Model proposed by Baidu Research and Silicon Valley AI Lab.

**Installing the required dependencies** : `$ pip install -r requirements.txt`

## Phase 1 : Data Generation and Augmentation

**Contents**:</br>

* [Augmentation](ds_utils/augmentation.py)
* [Data Generation](https://github.com/Stellarator-X/ConvoBot/blob/5e8af5538b3b9eb863606d9a0180b11efe284e8c/Programming%20Assignments/Speech%20Recognition/ds_utils/data_manip.py#L52)

These can be tested through the [test_phase1](test_phase1.py) script (results can be found [here](Results/Phase1.md)) :

```terminal
usage: test_phase1.py [-h] [--audio_file_path AUDIO_FILE_PATH]
                      [-a {spec,timeser}] [--dataset_path DATASET_PATH] [-d]

Tests the Data Generator and Data Augmentation

optional arguments:
  -h, --help            show this help message and exit

Augmentation Tests:
  --audio_file_path AUDIO_FILE_PATH
                        Path to the audio file
  -a {spec,timeser}, --augmentation {spec,timeser}
                        Type of augmentation : spec => on spectrogram, timeser
                        => on time series

Data Generation Tests:
  --dataset_path DATASET_PATH
                        Path to the dataset
  -d, --data_gen        Creates a Data Generator with the given dataset

Displays the time series/spectrogram of the augmented sample/Instantiates the
Data Generator
```

## References

* _Deep Speech 2: End-to-End Speech Recognition in English and Mandarin_
  * **Link** : [https://arxiv.org/abs/1512.02595]
  * **Author(s)/Organization** : Baidu Research – Silicon Valley AI Lab
  * **Tags** : Speech Recognition
  * **Published** : 8 Dec, 2015
* _SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition_
  * **Link** : [https://arxiv.org/abs/1904.08779]
  * **Authors** : Daniel S. Park , William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le
  * **Tags** : Speech Recognition, Data Augmentation
  * **Published** : 3 Dec, 2019

****
