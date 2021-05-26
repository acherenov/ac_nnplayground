import glob
from PIL import Image
import os

# filepaths
fp_in = "./pictures/nn_predictions_*.png"
fp_out = "./gifs/image.gif"

path = os.getcwd()
path = path + "/gifs"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)
