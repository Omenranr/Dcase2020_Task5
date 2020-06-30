#!/usr/bin/env bash

SONYC_UST_PATH=/home/server/Dcase/task5 ### Project path

yes | pip3 install -r requirements.txt

# Download dataset
mkdir -p $SONYC_UST_PATH/embeddings_ef_3ch_5s_train
mkdir -p $SONYC_UST_PATH/data
mkdir -p $SONYC_UST_PATH/output

pushd $SONYC_UST_PATH/data
wget https://zenodo.org/record/3693077/files/audio.tar.gz
wget https://zenodo.org/record/3693077/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/3693077/files/README.md

# Decompress audio
tar xf audio.tar.gz
rm audio.tar.gz
popd
