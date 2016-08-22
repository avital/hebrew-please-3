from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2, l1, l1l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

def make_model(optimizer):
    model = Sequential()

    L2_REGULARIZATION = 0.01
    L1_REGULARIZATION = 0
    INITIAL_DROPOUT = 0
    DROPOUT = 0
    FC_DROPOUT = 0.5
    GAUSSIAN_NOISE = 1.0

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 257, 320)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(32, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(DROPOUT))
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(64, 5, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 81, 1, W_regularizer=l1(L1_REGULARIZATION)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(AveragePooling2D(pool_size=(1, 2)))


    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(256, 1, 3, W_regularizer=l2(L2_REGULARIZATION)))

    model.add(Flatten())

    model.add(Dropout(FC_DROPOUT))
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(256, W_regularizer=l2(L2_REGULARIZATION)))

    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(256, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(2, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(Activation('softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
