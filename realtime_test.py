# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:24:25 2020

@author: pankaj.mishra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import numpy as np
import argparse
import imutils
import cv2
import model as mdl

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=False,default = './visualize/haarcascade_frontalface_default.xml',help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=False, default= './public_model_414_55.pt',help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
 

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])

model = mdl.Model(num_classes=len(EMOTIONS))
model.load_state_dict(torch.load(args["model"]))
model.eval()


# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])
    
# keep looping
while True:
# grab the current frame
    (grabbed, frame) = camera.read()
# if we are viewing a video and we did not grab a
# frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# initialize the canvas for the visualization, then clone
# the frame so we can draw on it
    canvas = np.zeros((220, 300, 3), dtype="uint8")
    frameClone = frame.copy()

# detect faces in the input frame, then clone the frame so that
# we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, 
                                      minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # ensure at least one face was found before continuing
    if len(rects) > 0:
# determine the largest face area
        rect = sorted(rects, reverse=True,
                      key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect
# extract the face ROI from the image, then pre-process
# it for the network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi/ 255.0
#        roi = img_to_array(roi)
#        roi = roi.permute(1,2,0).numpy() ## Changin the numpy array
        roi = np.expand_dims(roi, axis=0)
        
## Making prediction
        # make a prediction on the ROI, then lookup the class
        # label
        with torch.no_grad():
            preds = model(torch.as_tensor(roi,dtype=torch.float).unsqueeze(0)).cpu().numpy().reshape(-1).tolist()
        label = EMOTIONS[np.argmax(preds)]
        
        # loop over the labels + probabilities and draw them
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        # draw the label + probability bar on the canvas
        w = int(prob * 300)
        cv2.rectangle(canvas, (5, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
        # draw the label on the frame
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
        # show our classifications + probabilities
    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)
# if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()