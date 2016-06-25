# train7.py

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization

import numpy
from scipy import ndimage

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping

import os

model = Sequential()

L2_REGULARIZATION = 0.000
INITIAL_DROPOUT = 0.2
DROPOUT = 0.5

model.add(ZeroPadding2D((1, 1), input_shape=(1, 257, 320)))
model.add(Dropout(INITIAL_DROPOUT))

model.add(Convolution2D(4, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())

model.add(Convolution2D(4, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(4, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(4, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(4, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())


model.add(Flatten())

model.add(Dense(8))
model.add(Dropout(DROPOUT))
model.add(ELU())
model.add(BatchNormalization())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer=Adadelta(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)

data = None
labels = []

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data_from_samples(samples_dir):
    data = []
    snippets = os.listdir(samples_dir)

    for snippet_id in snippets:
        dir = '{0}/{1}'.format(samples_dir, snippet_id)
        image_matrix = ndimage.imread('{0}/spectrogram.png'.format(dir), flatten=True)
        image_tensor = numpy.expand_dims(image_matrix, axis=0)
        data.append(image_tensor)

    return numpy.stack(data)

data0 = load_data_from_samples('../process-videos/data/0')
labels0 = [0] * len(data0)
data1 = load_data_from_samples('../process-videos/data/1')
labels1 = [1] * len(data1)
data = numpy.concatenate((data0, data1))
labels = labels0 + labels1

val_data0 = load_data_from_samples('../process-videos/data/0-val')
val_labels0 = [0] * len(val_data0)
val_data1 = load_data_from_samples('../process-videos/data/1-val')
val_labels1 = [1] * len(val_data1)
val_data = numpy.concatenate((val_data0, val_data1))
val_labels = val_labels0 + val_labels1

model.fit(
    data,
    labels,
    nb_epoch=10000,
    batch_size=128,
    validation_data=(val_data, val_labels),
    callbacks=[
        ModelCheckpoint(filepath="/mnt/weights.{epoch:02d}-{val_acc:.2f}.hdf5"),
        EarlyStopping(patience=5000)
    ]
)



