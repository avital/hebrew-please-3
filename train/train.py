from model import make_model

import numpy
from scipy import ndimage
import random

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping

import os

def main():
    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

    data = None
    labels = []

    (data, labels) = load_from_labelled_dirs(
        '../process-videos/data/0',
        '../process-videos/data/1',
        bootstrap_resample=True
    )
    (val_data, val_labels) = load_from_labelled_dirs(
        '../process-videos/data/0-val',
        '../process-videos/data/1-val'
    )

    model.fit(
        data,
        labels,
        nb_epoch=10000,
        batch_size=128,
        validation_data=(val_data, val_labels),
        callbacks=[
            ModelCheckpoint(filepath="/mnt/weights.{epoch:02d}-{val_acc:.2f}.hdf5"),
            EarlyStopping(monitor='val_loss', patience=5)
        ]
    )

    model.save_weights('weights-{0}.hdf5'.format(random.choice(xrange(100, 1000))))

def load_from_labelled_dirs(dir_0, dir_1, bootstrap_resample=False):
    data0 = load_samples(dir_0, bootstrap_resample)
    labels0 = [0] * len(data0)

    data1 = load_samples(dir_1, bootstrap_resample)
    labels1 = [1] * len(data1)

    data = numpy.concatenate((data0, data1))
    labels = labels0 + labels1

    return (data, labels)

def load_samples(samples_dir, bootstrap_resample):
    data = []
    snippets = os.listdir(samples_dir)

    if bootstrap_resample:
        snippets = numpy.random.choice(snippets, (len(snippets)))

    for snippet_id in snippets:
        dir = '{0}/{1}'.format(samples_dir, snippet_id)
        image_matrix = ndimage.imread('{0}/spectrogram.png'.format(dir), flatten=True)
        image_tensor = numpy.expand_dims(image_matrix, axis=0)
        data.append(image_tensor)

    return numpy.stack(data)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()




