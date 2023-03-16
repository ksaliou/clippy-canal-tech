
from pathlib import Path
from pathlib import PurePath
import boto3
import cv2
import base64

def get_presigned_url_s3(file_path):
    """
    Get presigned url of a file coming from s3
    @param file_path: s3 file path (s3://...)
    @return: Presigned url
    """
    expriation_time = 7200
    s3 = boto3.client('s3')
    bucket, key = file_path.replace("s3://", "").split("/", 1)
    presign_url = s3.generate_presigned_url('get_object',
                                            Params={'Bucket': bucket, 'Key': key},
                                            ExpiresIn=expriation_time)
    return presign_url

def frame_to_tc(frame, fps) :
    minutes = int(frame / (fps * 60))
    seconds = int((frame % (fps * 60)) / fps)
    frames = int(frame % fps)
    return str(minutes) + ':' + str(seconds) + ':' + str(frames)

def image_to_base64(image):
    # encode image
    _, buffer = cv2.imencode('.jpg', image)
    jpg64 = base64.b64encode(buffer)
    jpg_str = jpg64.decode('utf-8')
    return jpg_str