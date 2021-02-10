# Transformer Applications

## Specializations - Machine Learning ― Supervised Learning

## Description

* This repository contains some Transformer Applications exercises

## Learning Objectives

**Understand:**

* How to use Transformers for Machine Translation
* How to write a custom train/test loop in Keras
* How to use Tensorflow Datasets

## TF Datasets
* For machine translation, we will be using the prepared Tensorflow Datasets ted_hrlr_translate/pt_to_en for English to Portuguese translation
```python
pip install --user tensorflow-datasets
```

To use this dataset, we will have to use the Tensorflow 2.0 compat within Tensorflow 1.15 and download the content:
```python
#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()
pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
```
## Dependencies
```
Python 3.6.12
numpy 1.16
tensorflow 1.15
```

## Repo content

* **Main Folder that contains all main of the following tasks:**

| Task | Description |
| --- | --- |
|**0. Dataset**| class Dataset that loads and preps a dataset for machine translation
|**1. Encode Tokens**| encodes a translation into tokens
|**2. TF Encode**| acts as a tensorflow wrapper for the encode instance method
|**3. Pipeline**| Update the class Dataset to set up the data pipeline
|**4. Create Masks**| creates all masks for training/validation
|**5. Train**| creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset

## Usage
* Clone the repo and execute the main files

## Author
- [Cristian G](https://github.com/cristian-fg)

## License
[MIT](https://choosealicense.com/licenses/mit/)
