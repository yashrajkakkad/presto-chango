# Music Identification Through Audio Fingerprinting
Identifies a song from a small recording (say 30 seconds). The song has to be a part of your created database, of course.

This project is essentially a simplified version of what Shazam does.
## The Idea
The flowchart on the right describes the series of steps.
<img src="https://github.com/yashrajkakkad/presto-chango/blob/master/flowchart.png?raw=true" align="right" height="440">


Below is a brief summary. For detailed explanation with analysis, check out our [Report](https://drive.google.com/open?id=1xbEC75FN3AIidWBd8bckgi4QeNJDdN-b).

We decimate the audio signal by a factor of 4 after passing it through a low pass filter (to smartly avoid [aliasing](https://en.wikipedia.org/wiki/Aliasing)). Thereafter, the signal is converted to frequency domain using the famous Fast Fourier Transform.

We take small chunks of the sample (roughly 0.3 seconds) and take the peak frequencies along a logarithmic scale. Those values are then associated with a hash value. We do so for all the songs and hence create a database.

We perform similar steps for the recorded sample. The answer is the song with the highest number of matches for a particular offset value.

## Requirements
- Python 3+.
- pip (package installer for Python). See [here](https://pip.pypa.io/en/stable/installing/) for installation.
- ffmpeg. See [here](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg) for installation.
- PortAudio. Only for Linux/OSX users. Check your distribution's repos for latest builds. Instead you can also build it from source, see [here](http://www.portaudio.com/download.html).

## Installation
- Install using pip (preferred in a virtual environment).
```sh
pip install Presto-Chango
```

## Usage
- On the first run, create your database by specifying the location of your songs directory.
```sh
presto-chango create-db <songs-directory>
```
- Identify a song by either recording in real time or using a pre-recorded sample.
```sh
# Record in real time
presto-chango identify

# Use a pre-recorded sample
presto-chango identify --file=samples/sample1.wav
```
- The algorithm returns the top five matches and the number of offsets that matched for each of them. Example
```sh
$ presto-chango identify --file="samples/sample_GAY.wav"
  Loading database
  .
  .
  .
  Database loaded

  Processing...

  Results:
  Kane Brown - Good as You (Official Music Video)_mS3TeZEp_PE.wav 41
  Katy Perry - Never Really Over (Official)_aEb5gNsmGJ8.wav 39
  Ed Sheeran - Perfect (Official Music Video)_2Vv-BfVoq4g.wav 37
  Cody Johnson - On My Way To You (Official Music Video)_RKUENGsDXBA.wav 24
  Jason Aldean - Rearview Town_WEUUvntknTI.wav 23
```

## Building the source code
- Clone the repository
```sh
git clone https://github.com/yashrajkakkad/presto-chango.git
cd presto-chango
```
- Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate
```
- Install the package. The `--editable` flag makes it so that we don't have to reinstall everytime we make some change.
```sh
pip install --editable .
```

## Testing
You can run the tester code if you're too lazy to record songs. It will cut random 30 second samples from songs and run the algorithm.
```sh
python tester.py
```

## References
- [How does Shazam work | Coding Geek](http://coding-geek.com/how-shazam-works/)
- [Creating Shazam in Java](https://royvanrijn.com/blog/2010/06/creating-shazam-in-java/)
