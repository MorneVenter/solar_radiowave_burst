import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import sys
import os

path = sys.argv[1]
if os.path.exists(path):
    files = path
    image = CallistoSpectrogram.read(files)
    plt.figure(figsize=(16,6))
    image.plot( cmap='inferno')
    plt.axis('equal')
    plt.title ('Spectogram')
    plt.savefig('outpt.png')
else:
    print('Invalid file.')
