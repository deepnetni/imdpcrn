# Samples

The `samples` directory contains the self recorded samples under far-end single talk scenario.

# Scripts to generate dataset

- The code to generate the training dataset is in `augment_util.py` which loads the settings in `aug.cfg`. 

You need to modify the `aug.cfg` file to specifiy the path of the DNS, the AEC challenge dataset, and the output directory before running the following commands to generate the training dataset:

```python
python augment_util.py
```

- The code to synthesis the recorded test dataset is in `concat_clean_speech.py`.

You can synthesis the test dataset with the following commands using our recorded FE and LibriSpeech samples:

```python
python concat_clean_speech.py --gene --sph_dir /path/to/librispeech_train/dev-clean --out_dir /path/to/output
```


# IMDPCRN API

First install the depencies from `requirements.txt`.

Then, our proposed IMDPCRN could be tested with the following commands:

```python
python run.py --mode file --lpb /path/to/lpb.wav --mic /path/to/mic.wav --out /path/to/out
python run.py --mode dir --src /path/to/blind_dataset --out /path/to/output_dir
```

The output echo-removed wav will be generated to the `--out` path if it is configured; otherwise, the output wav file will be created at the current directory.

