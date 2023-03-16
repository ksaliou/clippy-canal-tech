import argparse
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN
from PIL import Image
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from os.path import isfile, join, dirname, join
from collections import deque
from pathlib import PurePath

import cv2

from architecture import *

BUF_SIZE = 255
readFinished = False

FACE_DETECTION_CONFIDENCE = 0.95
#FACE_RECOGNITION_CONFIDENCE = 0.8

def create_recognition_models(embedings_path, facenet_path):
    # create the detector, using default weights
    face_detector = MTCNN()

    face_data_path = embedings_path
    facenet_model_path = facenet_path

    data = np.load(face_data_path)
    trainX, trainy = data['arr_0'], data['arr_1']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)

    # fit model
    recognition_model = SVC(decision_function_shape='ovo', kernel='linear', probability=True)
    recognition_model.fit(trainX, trainy)

    # Get image embedding
    
    facenet_model = InceptionResNetV2()
    facenet_model.load_weights(facenet_model_path)

    return (face_detector, facenet_model, recognition_model, out_encoder)


def extract_faces(frame, face_detector, required_size=(160, 160)):
    faces = list()
    # detect faces in the image
    results = face_detector.detect_faces(frame)
    # extract the bounding box from each faces
    for result in results:
        confidence = result['confidence']
        x1, y1, width, height = result['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = frame[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        faces.append((x1, y1, width, height, confidence, face_array))
    return faces

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0] 

def predict(faceEmb, model, out_encoder):
    # prediction for the face
    samples = np.expand_dims(faceEmb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    persons = list()
    for i in range(0, min(len(yhat_class), 3)):
        # get name
        class_index = yhat_class[i]
        class_probability = yhat_prob[i, class_index]
        predict_names = out_encoder.inverse_transform(yhat_class)
        print("Person detected " + predict_names[i] + ' : ' +  str(class_probability))
        persons.append({'name': predict_names[i], 'score': class_probability})
    return persons

def faces_recognition(frame, face_detector, facenet_model, recognition_model, out_encoder):
    facesData = extract_faces(frame, face_detector)
    faces = list()
    for (x, y, w, h, confidence, faceData) in facesData:
        if confidence >= FACE_DETECTION_CONFIDENCE:
            faceEmb = get_embedding(facenet_model, faceData)
            predictions = predict(faceEmb, recognition_model, out_encoder)
            position = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
            faces.append({'position': position, 'person': predictions})

    return faces

def parseFrame(frame, frameIdx):
    frame = cv2.resize(frame, (1024, 576)) 
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    h1 = cv2.normalize(h1, h1)
    return ((h1, frame, frameIdx))

def showFrame(frame, frameIdx, faces, out):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 1.2
    fontColor              = (0,0,255)
    thickness              = 2
    lineType               = 2

    # displayedFrame = frame
    # if(frameIdx < lastDetectionFrame + 50 and lastDetectionFrame > 0):
    # if(frameIdx < lastDetectionFrame + 50 and lastDetectionFrame > 0):
    #     cv2.putText(frame,
    #         shotStr, 
    #         (10,550), 
    #         font, 
    #         fontScale,
    #         (255,255,255),
    #         thickness,
    #         lineType)

    for face in faces:
        className = face['person'][0]['name']
        (x1, y1) = (int(face['position']['x']), int(face['position']['y']))
        (x2, y2) = (int(face['position']['x']) + int(face['position']['w']), int(face['position']['y']) + int(face['position']['h']))
        #print(str((x1, y1)) + " - " + str((x2, y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), fontColor, 2)
        cv2.putText(frame,
            className + " - " + str(round(face['person'][0]['score'], 2)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        bottomLeftCornerOfText = (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 40)

    sizedFrame = cv2.resize(frame,(1024,576))
    #cv2.imshow('frame', sizedFrame)
    out.write(sizedFrame)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True

def process_frames(file, embeddings_path, facenet_path):
    
    cap = cv2.VideoCapture(file)

    (face_detector, facenet_model, recognition_model, out_encoder) = create_recognition_models(embeddings_path, facenet_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('out_persons.m4v', fourcc, 25.0, (1024, 576))
    fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = int(fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alpha = 1.5
    window = deque(maxlen = window_size)
    window_frames = deque(maxlen = window_size)
    
    framesByIndex = {}

    # read first frame 
    ret, frame = cap.read()
    if not ret or frame is None:
        return
    (lastHistogram, lastFrame, lastIndex) = parseFrame(frame, 0)
    framesByIndex[lastIndex] = lastFrame
    frameIdx = 1
    faces = list()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frameIdx % 15 == 0:
            faces = faces_recognition(frame, face_detector, facenet_model, recognition_model, out_encoder)
        
        showFrame(frame, frameIdx, faces, out)

        frameIdx = frameIdx + 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def person_recognition(file, embeddings_path, facenet_path):
    start_time = time.time()
        
    print("starting processing frames : " + str((time.time() - start_time)))
    process_frames(file, embeddings_path, facenet_path)

    print("--- %s seconds ---" % (time.time() - start_time))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Clippy arguments.')
    parser.add_argument("input", help="Path of the input video file")
    parser.add_argument("embeddings", help="Path of the embeddings file")
    parser.add_argument("facenet", help="Path of the facenet file")
    args = parser.parse_args()

    # source = cv2.imread(args.input)
    # (session, return_tensors, classes) = create_object_detection_models()
    # (objects, bboxes) = get_objects(session, return_tensors, classes, source)
    # showImg(source, bboxes, classes)
    person_recognition(str(args.input), str(args.embeddings), str(args.facenet))
