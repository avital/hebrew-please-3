from model import make_model

import numpy
from scipy import ndimage

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
        '../process-videos/data/1'
    )
#    (data, labels) = resample(data, labels)

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
            EarlyStopping(monitor='val_loss', patience=1)
        ]
    )

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




