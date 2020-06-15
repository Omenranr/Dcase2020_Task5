#!/usr/bin/env bash

# Activate environment
#source activate sonyc-ust

# Extract embeddings

#pip3 install "tensorflow-gpu==2.0.0"
#pip3 install "tensorflow-gpu<1.14"

SONYC_UST_PATH=/home/server/Dcase/task5 ### Project path

pushd src

python3 data_create.py $SONYC_UST_PATH/data/audio/ $SONYC_UST_PATH/embeddings_ef_3ch_5s_train/

python3 classify.py $SONYC_UST_PATH/data/annotations_train.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/output baseline_fine \
  --label_mode fine --learning_rate 1e-4 --l2_reg 1e-4 \
  --dropout_size 0.4 --ef_mode 4 --num_epochs 8 \
  --emb_dir /home/server/Dcase/task5/embeddings_ef_3ch_5s_train/

python3 evaluate_predictions.py $SONYC_UST_PATH/output/baseline_fine/ $SONYC_UST_PATH/data/annotations_train.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml

popd
