import os
import random
import subprocess
import math

from utils import make_sure_path_exists

NUM_TRAINING_EXAMPLES=30000
NUM_VALIDATION_EXAMPLES=6000

VIDEOS_DATA_DIR='../download-videos/data/'

def main():
    make_training_examples()
    make_validation_examples()

def make_training_examples():
    for i in xrange(NUM_TRAINING_EXAMPLES):
        print "Training: {0}/{1}".format(i+1, NUM_TRAINING_EXAMPLES)
        label = random.choice([0, 1])
        example_dir = 'data/{0}/{1}'.format(label, i)
        make_sure_path_exists(example_dir)

        label_videos_dir = '{0}/{1}'.format(VIDEOS_DATA_DIR, label)
        videos = os.listdir(label_videos_dir)
        video = random.choice(videos)
        wav_file = '{0}/{1}/audio.wav'.format(label_videos_dir, video)

        random_segment_file = '{0}/segment.wav'.format(example_dir)
        cut_random_segment(wav_file, random_segment_file)

        stretched_segment_file = '{0}/stretched.wav'.format(example_dir)
        stretch(random_segment_file, stretched_segment_file)

        noise_file = '{0}/noise.wav'.format(example_dir)
        noisy_segment_file = '{0}/noisy.wav'.format(example_dir)
        add_random_noise(stretched_segment_file, noise_file, noisy_segment_file)

        normalized_segment_file = '{0}/normalized.wav'.format(example_dir)
        normalize(noisy_segment_file, normalized_segment_file)

        spectrogram_numpy_file = '{0}/spectrogram.npy'.format(example_dir)
        spectrogram_png_file = '{0}/spectrogram.png'.format(example_dir)
        make_spectrogram(normalized_segment_file, spectrogram_numpy_file, spectrogram_png_file)


def make_validation_examples():
    for i in xrange(NUM_VALIDATION_EXAMPLES):
        print "Validation: {0}/{1}".format(i+1, NUM_VALIDATION_EXAMPLES)
        label = random.choice([0, 1])
        example_dir = 'data/{0}-val/{1}'.format(label, i)
        make_sure_path_exists(example_dir)

        label_videos_dir = '{0}/{1}'.format(VIDEOS_DATA_DIR, label)
        videos = os.listdir(label_videos_dir)
        video = random.choice(videos)
        wav_file = '{0}/{1}/audio.wav'.format(label_videos_dir, video)

        random_segment_file = '{0}/segment.wav'.format(example_dir)
        cut_random_segment(wav_file, random_segment_file)

        normalized_segment_file = '{0}/normalized.wav'.format(example_dir)
        add_random_noise(random_segment_file, normalized_segment_file)

        spectrogram_numpy_file = '{0}/spectrogram.npy'.format(example_dir)
        spectrogram_image_file = '{0}/spectrogram.png'.format(example_dir)
        make_spectrogram(normalized_segment_file, spectrogram_numpy_file, spectrogram_image_file)

def cut_random_segment(in_wav_file, out_wav_file):
    # XXX!!! brittle if we change file format
    num_secs = os.path.getsize(in_wav_file) / 11025 / 2
    segment_secs = 2
    start_sec = random.uniform(0, num_secs - segment_secs)
    subprocess.check_call([
        'sox',
        in_wav_file,
        out_wav_file,
        'trim',
        str(start_sec),
        str(segment_secs),
    ])

def stretch(in_wav_file, out_wav_file):
    factor = math.exp(random.uniform(math.log(0.9), math.log(1.1)))
    subprocess.check_call([
        'sox',
        in_wav_file,
        out_wav_file,
        'tempo',
        '-s',
        str(factor),
        'trim',
        '0',
        '1.6',
    ])

def add_random_noise(in_wav_file, noise_wav_file, out_wav_file):
    subprocess.check_call([
        'sox',
        in_wav_file,
        noise_wav_file,
        'synth',
        'whitenoise',
        'vol',
        '0.02',
    ])
    subprocess.check_call([
        'sox',
        '-m',
        in_wav_file,
        noise_wav_file,
        out_wav_file,
    ])

def normalize(in_wav_file, out_wav_file):
    subprocess.check_call([
        'sox',
        '--norm',
        in_wav_file,
        out_wav_file,
    ])

def make_spectrogram(in_wav_file, out_numpy_file, out_png_file):
    subprocess.check_call([
        'sox',
        in_wav_file,
        '-n',
        'spectrogram',
        '-y', '257', # 65 FFT bins
        '-x', '257', # width
        '-r', # raw spectrogram
        '-o',
        out_png_file,
    ])

if __name__ == '__main__':
    main()
