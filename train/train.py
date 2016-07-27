from model import make_model

import sys
import numpy
from scipy import ndimage
import random
import socket

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard

import os

samples = {
  0: [num for num in os.listdir('../process-videos/data/0') if int(num) >= 1000],
  1: [num for num in os.listdir('../process-videos/data/1') if int(num) >= 1000]
}

print samples[0].length
print samples[1].length
print


def main():
    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

    (val_data, val_labels) = load_from_labelled_dirs(
        '../process-videos/data/0',
        '../process-videos/data/1',
        max_num=1000,
    )

    def val_data_generator():
        batch_size = 50
        cursor = 0
        while True:
            batch_data = val_data[cursor:cursor+batch_size]
            batch_labels = val_labels[cursor:cursor+batch_size]
            cursor = (cursor + batch_size) % len(val_data)
            yield (batch_data, batch_labels)

    def data_generator():
        batch_size = 128
        while True:
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                samples_dir = '../process-videos/data/{0}'.format(label)
                sample = random.choice(samples[label])
                sample_dir = '{0}/{1}'.format(samples_dir, sample)
                image_matrix = ndimage.imread('{0}/spectrogram.png'.format(sample_dir), flatten=True)
                image_tensor = numpy.expand_dims(image_matrix, axis=0)
                batch_data.append(image_tensor)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=100,
        validation_data=val_data_generator(),
        nb_val_samples=len(val_data),
        callbacks=[
            TensorBoard(log_dir='/mnt/nfs/logs-{0}'.format(socket.gethostname()),
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

    model.save_weights('weights.hdf5')

def load_from_labelled_dirs(dir_0, dir_1, max_num=99999999999):
    data0 = load_samples(dir_0, max_num)
    labels0 = [0] * len(data0)

    data1 = load_samples(dir_1, max_num)
    labels1 = [1] * len(data1)

    data = numpy.concatenate((data0, data1))
    labels = labels0 + labels1

    return (data, labels)

def load_samples(samples_dir, max_num):
    data = []
    snippets = os.listdir(samples_dir)

    for snippet_id in snippets:
        if int(snippet_id) < max_num:
            dir = '{0}/{1}'.format(samples_dir, snippet_id)
            image_matrix = ndimage.imread('{0}/spectrogram.png'.format(dir), flatten=True)
            image_tensor = numpy.expand_dims(image_matrix, axis=0)
            data.append(image_tensor)
        else:
            print('Skipping {0}'.format(snippet_id))

    return numpy.stack(data)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()



