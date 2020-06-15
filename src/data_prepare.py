import numpy as np
import keras
import librosa

NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_list, label_list, data_index,
                latitude_list, longitude_list, year_list, week_list, day_list, hour_list,
                num_classes, path, batch_size=32, dim=(32,32,32),
                shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        self.path = path

        self.latitude_list = latitude_list
        self.longitude_list = longitude_list
        self.year_list = year_list
        self.week_list = week_list
        self.day_list = day_list
        self.hour_list = hour_list

        self.data_list = data_list
        self.label_list = label_list
        self.data_index = data_index
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_index) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        data_list_temp = [self.data_index[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_index))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, data_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        X1 =[]
        X2 = []
        def one_hot(idx, num_items):
            return [(0.0 if n != idx else 1.0) for n in range(num_items)]

        # Generate data
        for i, ID in enumerate(data_list_temp):

            # Store sample
            X_emb = np.load(self.path+ self.data_list[ID][:-4] + '.npz')['embedding']

            num_frames = X_emb.shape[0]
            X_loc = np.array([[self.latitude_list[ID], self.longitude_list[ID]]] * num_frames)

            X_time = np.array(
                [one_hot(self.week_list[ID] - 1, NUM_WEEKS) \
                  + one_hot(self.day_list[ID], NUM_DAYS) \
                  + one_hot(self.hour_list[ID], NUM_HOURS)] * num_frames)

            X_added = np.concatenate((X_loc, X_time), axis=-1)

            X1.append(X_emb)
            X2.append(X_added)

            # Store class
            y[i,] = self.label_list[ID]


        X = [np.array(X1), np.array(X2)]
        return X, y
