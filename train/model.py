from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization

def make_model():
    model = Sequential()

    L2_REGULARIZATION = 0
    INITIAL_DROPOUT = 0
    DROPOUT = 0
    FC_DROPOUT = 0

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 257, 320)))
    model.add(Dropout(INITIAL_DROPOUT))

    model.add(Convolution2D(8, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(24, 5, 3, subsample=(3, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(96, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(96, 3, 3, subsample=(2, 2), W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Flatten())

    model.add(Dropout(FC_DROPOUT))
    model.add(Dense(96, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))
    model.add(Dense(24, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))
    model.add(Dense(1, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adadelta(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
