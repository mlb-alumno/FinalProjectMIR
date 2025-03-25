# Chord recognition with deep learning

This repository contains the code for the final project of the course "MIR - Music Information Retrieval" at the UPF, Barcerlona as part of the Master's degree in Sound and music Computing. 

The authors of the project are:
- Theo Fuhrmann
- Manuel Lallana
- Carlos Patiño

## Requirements

a conda enviroment is recomended

```console
conda create -n chord_recognition python=3.11
```

```console
conda activate chord_recognition
```

```console
pip install -r requirements.txt
``````

## Data

Ensure you have a folder structure in your working path as follows:

```
working
├── chords
│   ├── model_deep
│   │   ├── (here the final model will be stored)
│   ├── jazznet_test00.csv (a .csv file with the names of the .jams files you want for test)
│   ├── jazznet_train00.csv (a .csv file with the names of the .jams files you want for train)
│   ├── pump
│   │   ├── .npz files (the .npz files will be stored here after using the data pump notebook in new_notebooks/Data_pump.ipynb)
│   └── pump.pkl (the pump.pkl will be stored here after using the data pump notebook in new_notebooks/Data_pump.ipynb)
├── plots
│   ├── (final plots will be stored here)
├── jazznet (here the jazznet dataset will be stored)
│   ├── clean_dataset
│   │   ├── jams
│   │   │   ├── train
│   │   │   │   ├── jams files (the jams files will be stored here after using the data pump notebook in new_notebooks)
│   │   │   ├── test 
│   │   │   │   ├── jams files (the jams files will be stored here after using the data pump notebook in new_notebooks)
│   │   ├── audios
│   │   │   ├── train
│   │   │   │   ├── jams files (the jams files will be stored here after using the data pump notebook in new_notebooks)
│   │   │   ├── test 
│   │   │   │   ├── jams files (the jams files will be stored here after using the data pump notebook in new_notebooks)
│   │   ├── metadata 
│   │   │   ├── small.csv (a .csv file with the jezznet metadata)
```

## Data preparation

### Audios and jam files

The data pump takes pairs of same name .jams and .wav files, so you need to have the same number of jams and wav files in the jazznet/clean_dataset/jams and jazznet/clean_dataset/audios folders.


## Data pump    

Open the 01 ``notebooks/Data_pump.ipynb`` notebook and run the code. This will create the .npz files and the pump.pkl file in the chords folder.

## Training

Run:

```bash
cd code
```

and then

```bash
python train.py --epochs 2 --epoch-size 10 --working /Users/your_user/working/chords --reference-path /Users/your_user/working/jazznet/clean_dataset/jams/test/
```
Just to test if it runs, then turn up the epochs and epoch size for real training.

## Inference 

The ``notebooks/Training_results.ipynb`` and ``notebooks/Test_results.ipynb`` notebooks contain the results of the training and testing.

Run the ``new_notebooks/Predict_new_sound.ipynb`` notebook to do inference on your own piano audio file.


## References

- https://github.com/bmcfee/ismir2017_chords/tree/master
- https://github.com/bmcfee/pumpp
