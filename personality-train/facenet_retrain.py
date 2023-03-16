from os import listdir, environ
#environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import isdir,join
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from multiprocessing import Pool
import sys

# extract a single face from a given photograph

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    print('processing %s' % (filename))
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if(len(results) == 0):
        return None
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory


def load_faces(directory):
    faces = list()
    print("Loading faces in directory : " + directory)
    # enumerate files
    #with Pool(5) as p:
    #    p.map(extract_face, map(s => directory + s, listdir(directory)))
    i = 0
    for filename in listdir(directory):
        i += 1
        if directory.find('unknown') <= 0 and i > 10:
            break

        # path
        path = join(directory,filename)

        if isdir(path):
            subDirFaces = load_faces(path + "\\")
            faces.extend(subDirFaces)
        else:
            # get face
            face = extract_face(path)
            if(face is not None):
                # store
                faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory):
    print ('listing directories in ' + directory)
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + '/' + subdir + '/'
        print ('listing faces in ' + path)

        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' %
            (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# load train dataset
trainX, trainy = load_dataset(sys.argv[1])
print(trainX.shape, trainy.shape)
# load test dataset
# testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
# save arrays to one file in compressed format
savez_compressed(sys.argv[2], trainX, trainy)