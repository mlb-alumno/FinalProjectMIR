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

### Modify Jam files
If you need to change anything from the jams files, you can use the script:
``python jazznet_to_jams.py --working /Users/your_user/working/jazznet --csv_file /Users/your_user/working/jazznet/metadata/small.csv`` script to create the .jams files from the jazznet .csv, 

## Data pump    

Open the 01 ``new_notebooks/Data_pump.ipynb`` notebook and run the code. This will create the .npz files and the pump.pkl file in the chords folder.

## Training

Run:

```bash
cd code
```

and then

```bash
python train_jazznet.py --epochs 2 --epoch-size 10 --working /Users/your_user/working/chords --reference-path /Users/your_user/working/jazznet/clean_dataset/jams/test/
```
Just to test if it runs, then turn up the epochs and epoch size for real training.

## Inference 

Run the ``new_notebooks/Inference_notebook.ipynb`` notebook. This will do inference on an example.

## TODO:

- Evaluation at the end of the model training is now broken, something due to annotations being overlaped or something, but inference works!
- Update the evaluation notebook in notebooks/03 - Results.ipynb to output something cool.
- See how well it detects inversions.


## References

- https://github.com/bmcfee/ismir2017_chords/tree/master
- https://github.com/bmcfee/pumpp
