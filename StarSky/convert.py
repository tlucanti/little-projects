
import sys
from PIL import Image

with Image.open(sys.argv[1]) as im:
    im.save(sys.argv[1][:-3] + 'png')

