import errno
import os
import subprocess
import youtube_dl

from videos import VIDEOS

SAMPLE_FREQ = 11025

def main():
    for label in [0, 1]:
        for url in VIDEOS[label]:
            video_id = url[-11:]
            processed_base_dir = os.path.abspath('data/{0}'.format(video_id))
            make_sure_path_exists(processed_base_dir)
            downloaded_audio_file = '{0}/audio.unknown'.format(processed_base_dir)
            converted_audio_file = '{0}/audio.wav'.format(processed_base_dir)
            download_audio_from_youtube(url, downloaded_audio_file)
            convert_audio_to_wav(downloaded_audio_file, converted_audio_file)

def download_audio_from_youtube(url, downloaded_audio_file):
    print()
    print('Downloading audio from video ({0})...'.format(downloaded_audio_file))
    ydl = youtube_dl.YoutubeDL({
        'format': 'bestaudio',
        'outtmpl': downloaded_audio_file,
        'quiet': True,
        'writeinfojson': True
    })
    ydl.download([url])
    print('DONE.')

def convert_audio_to_wav(downloaded_audio_file, converted_audio_file):
    print()
    print('Converting to WAV ({0})...'.format(converted_audio_file))
    subprocess.check_call(["ffmpeg", "-y", "-i", downloaded_audio_file, "-ar", \
                           str(SAMPLE_FREQ), "-ac", "1", "-acodec", "pcm_s16le", converted_audio_file])
    print('DONE.')

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()


