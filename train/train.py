from model import make_model

import sys
import numpy
from scipy import ndimage
import random
import socket

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard

import os

samples = {
  0: [file for file in os.listdir('../data/training/v0/hebrew') if file.endswith('.spectrogram.png')],
  1: [file for file in os.listdir('../data/training/v0/english') if file.endswith('.spectrogram.png')],
}
num_samples = len(samples[0]) + len(samples[1])

val_samples = {
  0: [file for file in os.listdir('../data/validation/v0/hebrew') if file.endswith('.spectrogram.png')],
  1: [file for file in os.listdir('../data/validation/v0/english') if file.endswith('.spectrogram.png')],
}
num_val_samples = len(val_samples[0]) + len(val_samples[1])

def main():
    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

    def data_generator():
        batch_size = 128
        while True:
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                samples_dir = '../data/training/v0/{0}'.format('english' if label else 'hebrew')
                sample = random.choice(samples[label])
                spectrogram_file = '{0}/{1}'.format(samples_dir, sample)
                image_matrix = ndimage.imread(spectrogram_file, flatten=True)
                image_tensor = numpy.expand_dims(image_matrix, axis=0)
                batch_data.append(image_tensor)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    def val_data_generator():
        batch_size = 32
        while True:
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                samples_dir = '../data/validation/v0/{0}'.format('english' if label else 'hebrew')
                sample = random.choice(val_samples[label])
                spectrogram_file = '{0}/{1}'.format(samples_dir, sample)
                image_matrix = ndimage.imread(spectrogram_file, flatten=True)
                image_tensor = numpy.expand_dims(image_matrix, axis=0)
                batch_data.append(image_tensor)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=100,
        validation_data=val_data_generator(),
        nb_val_samples=128,
        callbacks=[
            TensorBoard(log_dir='/mnt/nfs/logs-{0}'.format(num_samples),
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

    model.save_weights('weights.hdf5', overwrite=True)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()



