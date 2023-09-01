# Scripts to generate dataset

- The code to generate the training dataset is in `augment_util.py` which loads the settings in `aug.cfg`. 

You need to modify the `aug.cfg` file to specifiy the path of the DNS, the AEC challenge dataset, and the output directory before running the following commands to generate the training dataset:

```python
python augment_util.py
```

- The code to synthesis the recorded test dataset is in `concat_clean_speech.py`.


# IMDPCRN API

First install the depencies from `requirements.txt`.

Then, our proposed IMDPCRN could be tested with the following commands:

```python
python run.py --lpb /path/to/lpb.wav --mic /path/to/mic.wav --out /path/to/out
```

The output echo-removed wav will be generated to the `--out` path if it is configured; otherwise, the output wav file will be created at the current directory.
