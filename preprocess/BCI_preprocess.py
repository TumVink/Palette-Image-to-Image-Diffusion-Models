import os
import numpy as np
from PIL import Image
import glob
#import imageio.v3 as iio

if __name__ == '__main__':
    root_dir = '/mnt/data/BCI/'
    split = 'train'
    format = 'HE'

    img_dir = os.path.join(root_dir,split,format)
    print(img_dir)
    files = glob.glob(img_dir+'/*.png')
    print(len(files))
    I = np.asarray(Image.open(files[0]))
    print(I.shape)