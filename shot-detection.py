import argparse
import time
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from requests.adapters import HTTPAdapter
from collections import deque
from pathlib import PurePath

from utils import *

BUF_SIZE = 255
readFinished = False

hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:,:,0] = h
hsv_map[:,:,1] = s
hsv_map[:,:,2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
hist_scale = 10
def set_scale(val):
    global hist_scale
    hist_scale = val

def parseFrame(frame, frameIdx):
    frame = cv2.resize(frame, (1024, 576)) 
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    h1 = cv2.normalize(h1, h1)
    return ((h1, frame, frameIdx))

def showFrame(frame, shotStr, frameIdx, lastDetectionFrame):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 0.8
    fontColor              = (0,0,255)
    thickness              = 2
    lineType               = 2

    displayedFrame = frame
    if(frameIdx < lastDetectionFrame + 50 and lastDetectionFrame > 0):
        cv2.putText(displayedFrame,
            shotStr, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    vis = getHsvImg(displayedFrame)
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #cvt = cv2.cvtColor(vis, cv2.COLOR_HSV2BGR)
    #print(vis.depth())
    cv2.imshow('movie', displayedFrame)
    cv2.imshow('hist', vis)
    #out.write(cvt)
    if cv2.waitKey(25) == ord('q'):
        return False
    return True

def getHsvImg(frame):
    small = cv2.pyrDown(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark = hsv[...,2] < 32
    hsv[dark] = 0
    h = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    h = np.clip(h*0.005*hist_scale, 0, 1)
    vis = hsv_map*h[:,:,np.newaxis] / 255.0
    #cv2.imshow('hist', vis)
    return vis


def process_frames(presigned_url):
    #cv2.imshow('hsv_map', hsv_map)
    #cv2.namedWindow('hist', 0)
    #cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)
    
    cap = cv2.VideoCapture(presigned_url)

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('outpy_overlay.m4v', fourcc, 25.0, (1024, 576), True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = int(fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alpha = 1.5
    window = deque(maxlen = window_size)
    window_frames = deque(maxlen = window_size)
    
    framesByIndex = {}
    shotId = 0
    last_boundary = 0
    shotFoundStr = ""

    # read first frame 
    ret, frame = cap.read()
    if not ret or frame is None:
        return
    (lastHistogram, lastFrame, lastIndex) = parseFrame(frame, 0)
    framesByIndex[lastIndex] = lastFrame
    frameIdx = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        (newHist, newFrame, newIndex) = parseFrame(frame, frameIdx)

        framesByIndex[newIndex] = newFrame
        window_frames.append((newIndex, newFrame))
        
        # compute histogram of the previous frame
        cmpHist = cv2.compareHist(lastHistogram, newHist, cv2.HISTCMP_INTERSECT)

        # add the histograme to the window
        window.append(cmpHist)

        # if the window is full, process the frame data
        if(len(window) == window_size):
            # the frame processed is the frame in the middle of the window
            processedFrame = newIndex - math.floor(window_size / 2) - 1
            mid = window[math.floor(window_size / 2) - 1]
            sortedWin = np.sort(window)
            m1 = sortedWin[0]
            m2 = sortedWin[1]
            mu = np.mean(sortedWin)
            sigma = np.std(sortedWin)

            if((frameIdx - last_boundary) > window_size):
                if mid == m1 and mid <= (alpha * m2) and mu >= (alpha * m1):
                    # print(np.array2string(sortedWin))
                    # print("mid=" + str(mid) + " m2=" + str(m2) +" mu=" + str(mu))
                    shot_type = 'hard-cut'
                elif mid <= (mu * (1 - sigma) + sigma ** 2) and sigma > 0 :
                    shot_type = 'dissolve' 
                else:
                    shot_type = 'none'
                
                # shot found
                if shot_type != 'none' and (processedFrame - last_boundary) > 30:
                    shotId += 1
                    key = math.floor((last_boundary + processedFrame) / 2)

                    start_tc = frame_to_tc(last_boundary, fps)
                    end_tc = frame_to_tc(processedFrame, fps)
                    print("Shot found between " + start_tc + ' and ' +  end_tc + ' - Key frame : ' + frame_to_tc(key, fps) + ' - ' + shot_type)     
                    shotFoundStr = "Shot found between " + start_tc + ' and ' +  end_tc + ' : ' + shot_type
                    thumb = framesByIndex[key]

                    #cv2.imwrite("shots/frame%d.jpg" % shotId, thumb)
                    last_boundary = processedFrame + 1
                    framesByIndex.clear()
                    for (wIdx, wFrm) in window_frames:
                        framesByIndex[wIdx] = wFrm

            showFrame(newFrame, shotFoundStr, frameIdx, last_boundary)

        # replace last values
        lastHistogram = newHist
        lastFrame = newFrame
        lastIndex = newIndex

        frameIdx = frameIdx + 1

    cap.release()
    #out.release()
    cv2.destroyAllWindows()

def shot_detection(file):
    start_time = time.time()

    print("starting getting url : " + str((time.time() - start_time)))
    if "s3" in file:
        url = get_presigned_url_s3(file)
    else:
        url = file
        
    print("starting processing frames : " + str((time.time() - start_time)))
    process_frames(url)

    print("--- %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Clippy arguments.')
    parser.add_argument("input", help="Path of the input video file")
    args = parser.parse_args()

    shot_detection(str(args.input))
