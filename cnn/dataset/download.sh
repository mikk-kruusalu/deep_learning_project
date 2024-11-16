#!/bin/bash
curl -L -o ./CNN/dataset/archive.zip \
https://www.kaggle.com/api/v1/datasets/download/bilalakgz/brain-tumor-mri-dataset

unzip -q -u ./CNN/dataset/archive.zip -d ./CNN/data/
