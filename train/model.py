from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization

def make_model():
    model = Sequential()

    L2_REGULARIZATION = 0
    INITIAL_DROPOUT = 0
    DROPOUT = 0
    FC_DROPOUT = 0.5
    NOISE = 0.03

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 257, 320)))
    model.add(GaussianNoise(NOISE))

    model.add(Convolution2D(8, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(GaussianNoise(NOISE))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(24, 5, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(GaussianNoise(NOISE))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution2D(48, 81, 1))

    model.add(Convolution2D(48, 1, 3))
    model.add(AveragePooling2D(pool_size=(1, 2)))


    model.add(Convolution2D(48, 1, 3))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(48, 1, 3))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(48, 1, 3))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(48, 1, 3))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(48, 1, 3))

    model.add(Flatten())

    model.add(Dropout(FC_DROPOUT))
    model.add(Dense(32, W_regularizer=l2(L2_REGULARIZATION)))

    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))
    model.add(Dense(24, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))

    model.add(Dense(1, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
