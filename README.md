DCASE 2020 Challenge: Task 5 - JHKim
=====================================

Fast run
------------

1. Check the SONYC_UST_PATH in "download_data.sh" and "train_start.sh"

2. run ./dwonload_data.sh

3. run ./train_start.sh

Requirements
------------
- pip3 install -r requirements.txt


information
------------

1. Data Prepare shell
    - ./download_data.sh

2. Main Sequence shell
    - ./train_start.sh

3. Prepare the feature for train
    - python3 src/feature_extraction.py \[audio_folder_path] \[feature_output_dir]
 
4. Train, create Model and predict validate set
    - python3 src/classify.py \[annotation_path] \[taxonomy_path] \[output_dir] \[--emb_dir] \[--ef_mode] \[--dropout_size] \[--learning_rate] \[--l2_reg] \[--batch_size] \[--num_epochs] \[--patience] \[--random-state]

5. Evaluate the predicted validate set
    - python3 src/evaluate_predictions.py \[output_dir] \[annotation_path] \[taxonomy_path]

Result
------------
Best result of task5 in our model 
- Fine : best_evaluation_fine.json
- Coarse : best_evaluation_coarse.json


