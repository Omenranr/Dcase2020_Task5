DCASE 2020 Challenge: Task 5 - JHKim
=====================================

Quick Start
------------

Check the SONYC_UST_PATH in "download_data.sh" and "train_start.sh"

`run ./dwonload_data.sh`

`run ./train_start.sh`

Requirements
------------
`pip3 install -r requirements.txt`


Information
------------

Data Prepare shell

`./download_data.sh`

Main Sequence shell
    
`./train_start.sh`

Prepare the feature for train

`python3 src/feature_extraction.py [audio_folder_path] [feature_output_dir]`
 
Train, create Model and predict validate set

`python3 src/classify.py [annotation_path] [taxonomy_path] [output_dir] [--emb_dir] [--ef_mode] [--dropout_size] [--learning_rate] [--l2_reg] [--batch_size] [--num_epochs] [--patience] [--random-state]`

Evaluate the predicted validate set

`python3 src/evaluate_predictions.py [output_dir] [annotation_path] [taxonomy_path]`

Result
------------
Best result of task5 in our model 

`Fine : best_evaluation_fine.json`

`Coarse : best_evaluation_coarse.json`


