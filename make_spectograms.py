
import matplotlib.pyplot as plt
from radiospectra.sources.callisto import CallistoSpectrogram
import numpy


files = 'type3.fit.gz'
image = CallistoSpectrogram.read(files)
nobg = plt.figure(figsize=(16,6))
nobg = image.subtract_bg()


nobg.plot(vmin=12, vmax = 255, cmap='inferno')
plt.axis('equal')
plt.title ('Type 3')
plt.savefig('output.png')
plt.show()
