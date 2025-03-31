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
├── model
│   ├── model_deep
│   │   ├── (here the final model will be stored)
│   ├── test00.csv (a .csv file with the names of the .jams files you want for test)
│   ├── train00.csv (a .csv file with the names of the .jams files you want for train)
│   ├── pump
│   │   ├── .npz files (the .npz files will be stored here after using the data pump notebook in new_notebooks/Data_pump.ipynb)
│   └── pump.pkl (the pump.pkl will be stored here after using the data pump notebook in new_notebooks/Data_pump.ipynb)
```

## Data preparation

### Audios and jam files

The data pump takes pairs of same name .jams and .wav files, so you need to have the same number of jams and wav files in the jazznet/clean_dataset/jams and jazznet/clean_dataset/audios folders.

## Data pump    

Open the 01 ``new_notebooks/Data_pump.ipynb`` notebook and run the code. This will create the .npz files and the pump.pkl file in the model folder.

## Training

1. Run:

```bash
cd code
```

2. Generate the train/test splits running the generate_train_test_csv.py script:

```bash
python generate_train_test_csv.py --folder "JAMS_FOLDER_PATH" --output_path "WORKING_MODEL_PATH" --train_ratio 0.8 --splits 5
```

3. Run the train.py script

```bash
python train.py --epochs 2 --epoch-size 10 --working /Users/your_user/working/chords --reference-path DATASET_ANNOTATIONS_PATH
```

Just to test if it runs, then turn up the epochs and epoch size for real training.

## Inference 

Run the ``notebooks/Predict_new_sound.ipynb`` notebook. This will do inference on an example.

## References

- https://github.com/bmcfee/ismir2017_chords/tree/master
- https://github.com/bmcfee/pumpp
