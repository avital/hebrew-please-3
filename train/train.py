from model import make_model

import sys
import numpy
from scipy import ndimage
import random

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping
from make_training_example import make_training_example

import os

def main():
    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

#    (val_data, val_labels) = load_from_labelled_dirs(
#        '../process-videos/data/0-val',
#        '../process-videos/data/1-val'
#    )

    def data_generator():
        batch_size = 128
        while True:
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                image_spectrogram = make_training_example(label, augment=True)
                image_matrix = ndimage.imread(image_spectrogram, flatten=True)
                image_tensor = numpy.expand_dims(image_matrix, axis=0)
                batch_data.append(image_tensor)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=10000,
#        validation_data=(val_data, val_labels),
#        validation_split=0.8,
#        callbacks=[
#           EarlyStopping(monitor='val_acc')
#        ]
    )

    model.save_weights('weights.hdf5')

def load_from_labelled_dirs(dir_0, dir_1):
    data0 = load_samples(dir_0)
    labels0 = [0] * len(data0)

    data1 = load_samples(dir_1)
    labels1 = [1] * len(data1)

    data = numpy.concatenate((data0, data1))
    labels = labels0 + labels1

    return (data, labels)

def load_samples(samples_dir):
    data = []
    snippets = os.listdir(samples_dir)

    for snippet_id in snippets:
        dir = '{0}/{1}'.format(samples_dir, snippet_id)
        image_matrix = ndimage.imread('{0}/spectrogram.png'.format(dir), flatten=True)
        image_tensor = numpy.expand_dims(image_matrix, axis=0)
        data.append(image_tensor)

    return numpy.stack(data)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()




