# calculate a face embedding for each face in the dataset using facenet
from architecture import * 

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from numpy import concatenate
#from keras.models import load_model
import tensorflow as tf

import sys

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load(sys.argv[1])
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)
# load the facenet model

model = InceptionResNetV2()
model.load_weights('facenet_keras_weights.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

#unknown_data = load(sys.argv[1])
#finalX = concatenate((unknown_data['arr_0'], newTrainX))
#finalY = concatenate((unknown_data['arr_1'], trainy))
finalX = newTrainX
finalY = trainy
# save arrays to one file in compressed format
savez_compressed(sys.argv[2], finalX, finalY)