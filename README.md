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
├── annotations
│   └── your .jams files
├── audio
│   └── tour .mp3 files
├── chords
│   ├── model_deep
│   │   ├── (here the final model will be stored)
│   ├── new_test00.csv (a .csv file with the names of the .jams files you want for test)
│   ├── new_train00.csv (a .csv file with the names of the .jams files you want for train)
│   ├── pump
│   │   ├── .npz files (the .npz files will be stored here after using the 01 data pump notebook)
│   └── pump.pkl (the pump.pkl will be stored here after using the 01 data pump notebook)
├── plots
│   ├── (final plots will be stored here)
├── reference_annotations
│   └── put here the .jams for testing
```

## Data preparation

### Audios
put your mp3 files in the audio folder inside your working path

### Jam files
run the ``code/create_jam_files.py`` script to create an example .jam file, put that file in the annotations folder inside your working path

TODO: load jams and mp3 from jazznet

## Data pump    

Open the 01 ``new_notebooks/Data_pump.ipynb`` notebook and run the code. This will create the .npz files and the pump.pkl file in the chords folder.

## Training

Run:

```bash
cd code
```

and then

```bash
python train_deep.py --epochs 2 --epoch-size 10 --working /Users/your_user/working/chords
```
Just to test if it runs, then turn up the epochs and epoch size for real training.

## Evaluation

Run the new/notebook/03 - Results.ipynb notebook. This will create the final plots.

## References

- https://github.com/bmcfee/ismir2017_chords/tree/master
- https://github.com/bmcfee/pumpp
