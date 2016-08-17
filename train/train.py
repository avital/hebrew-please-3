from model import make_model

import sys
import numpy
from scipy import ndimage
import random
import socket

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard

import os

samples = {
  samples_dir: [file for file in os.listdir(samples_dir) if file.endswith('.spectrogram.png')]
    for samples_dir in ['../data/nn/v0/train/english', '../data/nn/v0/train/english-avital', '../data/nn/v0/train/hebrew', '../data/nn/v0/train/hebrew-avital', '../data/nn/v0/2nd-val/hebrew', '../data/nn/v0/2nd-val/english']
}

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
                samples_dir = '../data/nn/v0/train/{0}'.format('english' if label else 'hebrew')
                if random.choice([1]):
                    samples_dir = samples_dir + '-avital'
                sample = random.choice(samples[samples_dir])
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
                samples_dir = '../data/nn/v0/2nd-val/{0}'.format('english' if label else 'hebrew')
                sample = random.choice(samples[samples_dir])
                spectrogram_file = '{0}/{1}'.format(samples_dir, sample)
                image_matrix = ndimage.imread(spectrogram_file, flatten=True)
                image_tensor = numpy.expand_dims(image_matrix, axis=0)
                batch_data.append(image_tensor)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=300,
        validation_data=val_data_generator(),
        nb_val_samples=128,
        callbacks=[
            TensorBoard(log_dir='/mnt/nfs/logs-v1-wider--with-avital-3-training-files-only-2ndval-0x30000',
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

    model.save_weights('weights.hdf5', overwrite=True)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()



