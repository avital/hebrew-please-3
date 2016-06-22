import os
import random

NUM_TRAINING_EXAMPLES=30000
NUM_VALIDATION_EXAMPLES=6000

def main():
    make_training_examples()
    make_validation_examples()

def make_training_examples():
    for i in xrange(NUM_TRAINING_EXAMPLES):
        label = random.choice([0, 1])
        videos = os.listdir('../download-videos/data/{0}'.format(label))
        video = random.choice(videos)
        print video

def make_validation_examples():
    for i in xrange(NUM_VALIDATION_EXAMPLES):
        label = random.choice([0, 1])
        videos = os.listdir('../download-videos/data/{0}'.format(label))
        video = random.choice(videos)

if __name__ == '__main__':
    main()
