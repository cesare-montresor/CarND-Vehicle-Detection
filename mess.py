import utils as u
import dataset as d
import glob

path = './datasources/vehicles/object-dataset/*.png'
img_paths = glob.glob(path)
for img_path in img_paths:
    img = u.loadImage(img_path)
    aug = d.randomHue(img)
    u.showImages( (img,aug) )


