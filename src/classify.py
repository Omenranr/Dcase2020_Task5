import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml
import random
import pickle as pk

import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, TimeDistributed ,Dropout, BatchNormalization
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from autopool import AutoPool1D
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import Model
# Generators
from keras.models import Sequential
from data_prepare import DataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)
random.seed(0)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

## HELPERS

def load_embeddings(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory
    Parameters
    ----------
    file_list
    emb_dir
    Returns
    -------
    embeddings
    ignore_idxs
    """
    embeddings = []
    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npz')
        embeddings.append(np.load(emb_path)['embedding'])

    return embeddings


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets
    Parameters
    ----------
    annotation_data
    Returns
    -------
    train_idxs
    valid_idxs
    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename', 'annotator_id']]\
                          .groupby(by=['split', 'audio_filename'], as_index=False)\
                          .min()\
                          .sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []

    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        elif row['split'] == 'validate' and row['annotator_id'] <= 0:
            # For validation examples, only use verified annotations
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels
    Parameters
    ----------
    annotation_data
    labels
    Returns
    -------
    targets
    """
    file_list = annotation_data['audio_filename'].unique().tolist()
    count_dict = {fname: {label: 0 for label in labels} for fname in file_list}

    for _, row in annotation_data.iterrows():
        fname = row['audio_filename']
        split = row['split']
        ann_id = row['annotator_id']

        # For training set, only use crowdsourced annotations
        if split == "train" and ann_id <= 0:
            continue

        # For validate and test sets, only use the verified annotation
        if split != "train" and ann_id != 0:
            continue

        for label in labels:
            count_dict[fname][label] += row[label + '_presence']

    targets = np.array([[1.0 if count_dict[fname][label] > 0 else 0.0 for label in labels]
                        for fname in file_list])

    return targets


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    Courtesy of https://stackoverflow.com/a/42797620
    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


## MODEL CONSTRUCTION
def construct_mlp(input_size, num_classes, num_frames,
                  dropout_size=0.5, ef_mode=4, l2_reg=1e-5):
    """
    Construct a MLP model for urban sound tagging.
    Parameters
    ----------
    num_frames
    input_size
    num_classes
    dropout_size
    ef_mode
    l2_reg
    Returns
    -------
    model
    """

    # Add hidden layers
    from keras.layers import Flatten, Conv1D, Conv2D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Concatenate, GlobalAveragePooling2D, LeakyReLU

    import efficientnet.keras as efn

    if ef_mode == 0:
        base_model = efn.EfficientNetB0(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 1:
        base_model = efn.EfficientNetB1(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 2:
        base_model = efn.EfficientNetB2(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 3:
        base_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 4:
        base_model = efn.EfficientNetB4(weights='noisy-student', include_top=False, pooling='avg')  #imagenet or weights='noisy-student'
    elif ef_mode == 5:
        base_model = efn.EfficientNetB5(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 6:
        base_model = efn.EfficientNetB6(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 7:
        base_model = efn.EfficientNetB7(weights='noisy-student', include_top=False, pooling='avg')

    input1 = Input(shape=input_size, dtype='float32', name='input')
    input2 = Input(shape=(num_frames,85), dtype='float32', name='input2') #1621
    y = TimeDistributed(base_model)(input1)
    y = TimeDistributed(Dropout(dropout_size))(y)
    y = Concatenate()([y, input2])
    y = TimeDistributed(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg)))(y)
    y = AutoPool1D(axis=1, name='output')(y)

    m = Model(inputs=[input1, input2], outputs=y)
    m.summary()
    m.name = 'urban_sound_classifier'

    return m

## DATA PREPARATION
def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]


def train_model(base_model, training_generator, validation_generator, output_dir,
                loss=None, batch_size=64, num_epochs=100, patience=20,
                learning_rate=1e-4):
    """
    Train a model with the given data.
    Parameters
    ----------
    model
    X_train
    y_train
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate
    Returns
    -------
    history
    """

    if loss is None:
        loss = 'binary_crossentropy'

    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')

    cb.append(keras.callbacks.ModelCheckpoint(output_dir + '/{epoch:02d}-{val_loss:.2f}_model_best.h5', verbose = 1,
                                              save_weights_only=True,
                                              save_best_only=False,
                                              monitor='val_loss')) #val_loss

    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', verbose = 1,
                                            patience=patience))

    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    model = ModelMGPU(base_model, gpus=2)
    model.compile(Adam(lr=learning_rate), loss=loss)

    history = model.fit_generator(generator=training_generator, validation_data=validation_generator,# steps_per_epoch=846,
        epochs=num_epochs, callbacks=cb, verbose=1, shuffle=False, use_multiprocessing=True, workers=8)

    return history


## MODEL TRAINING
def train(annotation_path, taxonomy_path, emb_dir, output_dir, exp_id,
          label_mode="fine", batch_size=64, num_epochs=1000,
          patience=20, learning_rate=1e-4, dropout_size=0.5,
          ef_mode=4, l2_reg=1e-5, standardize=True,
          timestamp=None, random_state=0):
    """
    Train and evaluate a MIL MLP model.
    Parameters
    ----------
    annotation_path
    emb_dir
    output_dir
    label_mode
    batch_size
    num_epochs
    patience
    learning_rate
    dropout_size
    l2_reg
    standardize
    timestamp
    random_state
    Returns
    -------
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    annotation_data_trunc = annotation_data[['audio_filename',
                                             'latitude',
                                             'longitude',
                                             'year',
                                             'week',
                                             'day',
                                             'hour']].drop_duplicates()
    file_list = annotation_data_trunc['audio_filename'].to_list()
    latitude_list = annotation_data_trunc['latitude'].to_list()
    longitude_list = annotation_data_trunc['longitude'].to_list()

    year_list = annotation_data_trunc['year'].to_list()  #### added
    week_list = annotation_data_trunc['week'].to_list()
    day_list = annotation_data_trunc['day'].to_list()
    hour_list = annotation_data_trunc['hour'].to_list()

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                          if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    print("* Preparing training data.")

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, valid_file_idxs = get_subset_split(annotation_data)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)
    print(np.shape(target_list[0]), np.shape(labels))

    embeddings = load_embeddings(file_list[:1], emb_dir)

    print(np.shape(embeddings[0]))

    dim = np.shape(embeddings[0]) #(11,597)
    num_frames = len(embeddings[0])
    print(num_frames)

    params = {'dim': dim,
    'batch_size': batch_size,
    'shuffle': True}

    scaler = None
    training_generator = DataGenerator(file_list, target_list, train_file_idxs,
                                latitude_list, longitude_list, year_list, week_list, day_list, hour_list,
                                len(target_list[0]), emb_dir, **params)
    validation_generator = DataGenerator(file_list, target_list, valid_file_idxs,
                                latitude_list, longitude_list, year_list, week_list, day_list, hour_list,
                                len(target_list[0]), emb_dir, **params)

    model = construct_mlp(dim,
                          num_classes, num_frames,
                          ef_mode=ef_mode,
                          dropout_size=dropout_size,
                          l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    if scaler is not None:
        scaler_path = os.path.join(results_dir, 'stdizer.pkl')
        with open(scaler_path, 'wb') as f:
            pk.dump(scaler, f)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
        loss_func = masked_loss
    else:
        loss_func = None

    training = False
    prediction = True

    if training == True:
        #history = train_model(model, X_train, y_train, X_valid, y_valid,
        #                          results_dir, loss=loss_func,
        #                      batch_size=batch_size, num_epochs=num_epochs,
        #                      patience=patience, learning_rate=learning_rate)
        history = train_model(model, training_generator, validation_generator,
                                  results_dir, loss=loss_func,
                              batch_size=batch_size, num_epochs=num_epochs,
                              patience=patience, learning_rate=learning_rate)

    # Reload checkpointed file
    if prediction == True:

        model = construct_mlp(dim,
                              num_classes, num_frames,
                              ef_mode=ef_mode,
                              dropout_size=dropout_size,
                              l2_reg=l2_reg)

        #model = multi_gpu_model(model, gpus=2)

        #model.summary()

        params = {'dim': dim,
        'batch_size': 1,
        'shuffle': False}
        scaler = None
        training_generator = DataGenerator(file_list, target_list, train_file_idxs,
                                    latitude_list, longitude_list, year_list, week_list, day_list, hour_list,
                                    len(target_list[0]), emb_dir, **params)
        validation_generator = DataGenerator(file_list, target_list, valid_file_idxs,
                                    latitude_list, longitude_list, year_list, week_list, day_list, hour_list,
                                    len(target_list[0]), emb_dir, **params)


        out_dir = os.listdir(results_dir)
        out_dir.sort()
        count = 0

        for i in out_dir:
            if i[-2:] == "h5":
                from keras.models import load_model
                from keras.models import model_from_json


                model_weight_file = os.path.join(results_dir, i)
                model.load_weights(model_weight_file)

                print("* Saving model predictions.")
                results = {}
                results['train'] = model.predict_generator(training_generator, use_multiprocessing=True, workers=8).tolist()
                results['validate'] = model.predict_generator(validation_generator, use_multiprocessing=True, workers=8).tolist()

                results_path = os.path.join(results_dir, "results.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                generate_output_file(results['validate'], valid_file_idxs, results_dir,
                                     file_list, label_mode, taxonomy, count)

                print(np.shape(results['validate']), np.shape(valid_file_idxs))

                count = count + 1


def generate_output_file(y_pred, file_idxs, results_dir, file_list, label_mode, taxonomy, count):
    """
    Write the output file containing model predictions
    Parameters
    ----------
    y_pred
    file_idxs
    results_dir
    file_list
    label_mode
    taxonomy
    Returns
    -------
    """
    output_path = os.path.join(results_dir, "output" + str(count) + ".csv")
    file_list = [file_list[idx] for idx in file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--emb_dir", type=str, default="/home/server/바탕화면/소스코드/연구코드/Dcase/task5/embeddings2_ef_3ch_5s/")
    parser.add_argument("--dropout_size", type=float, default=0.5)    # keep_prob 1.14    # rate > 2.x
    parser.add_argument("--ef_mode", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4) ## batch 64 1e-3, batch 32 1e-4
    parser.add_argument("--l2_reg", type=float, default=1e-4) # batch 8 1e-4 epoch4,  batch 4 1e-5
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(args.annotation_path,
          args.taxonomy_path,
          args.emb_dir,
          args.output_dir,
          args.exp_id,
          label_mode=args.label_mode,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          patience=args.patience,
          learning_rate=args.learning_rate,
          dropout_size=args.dropout_size,
          ef_mode=args.ef_mode,
          l2_reg=args.l2_reg,
          standardize=(not args.no_standardize),
          timestamp=timestamp,
          random_state=args.random_state)
