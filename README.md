# Music Identification Through Audio Fingerprinting
Identifies a song from a small recording (say 30 seconds). The song has to be a part of your created database, of course.

This project is essentially a simplified version of what Shazam does.
## The Idea
The below flowchart describes the series of steps:

![Flowchart](https://github.com/yashrajkakkad/presto-chango/blob/master/flowchart.png?raw=true)

Below is a brief summary. For detailed explanation with analysis, check out our [Report](https://drive.google.com/open?id=1xbEC75FN3AIidWBd8bckgi4QeNJDdN-b).

We decimate the audio signal by a factor of 4 after passing it through a low pass filter (to smartly avoid [aliasing](https://en.wikipedia.org/wiki/Aliasing)). Thereafter, the signal is converted to frequency domain using the famous Fast Fourier Transform.

We take small chunks of the sample (roughly 0.3 seconds) and take the peak frequencies along a logarithmic scale. Those values are then associated with a hash value. We do so for all the songs and hence create a database.

We perform similar steps for the recorded sample. The answer is the song with the highest number of matches for a particular offset value.

## Usage
- Clone the repository
```sh
git clone https://github.com/yashrajkakkad/presto-chango.git
```
- Download [FFmpeg](https://www.ffmpeg.org/download.html) and [PortAudio](http://www.portaudio.com/download.html).
- Install the required packages. Using a virtual environment is recommended.
```sh
pip install -r requirements.txt
```
- Create "Songs" directory and copy your favourite songs there. (They should strictly be in .wav format)
```sh
mkdir Songs
```
- Create a database from your Songs. It will be serialized using pickle.
```sh
python database.py
```
- Run the project. It will prompt you to play the song, which will be recorded via your mic. It will give you the 5 best matches with the number of matched values. 
```sh
python app.py
```

## Testing
You can run the tester code if you're too lazy to record songs. It will cut random 30 second samples from songs and run the algorithm.
```sh
python tester.py
```

## References
- [How does Shazam work | Coding Geek](http://coding-geek.com/how-shazam-works/)
- [Creating Shazam in Java](https://royvanrijn.com/blog/2010/06/creating-shazam-in-java/)
