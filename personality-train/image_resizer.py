import os
from os import listdir, mkdir
from os.path import isdir, isfile, join
from PIL import Image

onlydirs = [f for f in listdir('temp') if isdir(join('temp', f))]

for dir in onlydirs:
    images = [f for f in listdir(join('temp', dir))]

    if not os.path.exists(join('cleaned', dir)):
        mkdir(join('cleaned', dir))

    for image in images:
        imgData = Image.open(join('temp', dir, image))
        new_image = imgData.resize((244, 244)).convert("RGB")
        newPath = join('cleaned', dir, image)

        if os.path.exists(newPath):
            os.remove(newPath)

        new_image.save(newPath)