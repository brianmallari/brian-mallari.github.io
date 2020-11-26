# Project: Face Detection
# Author: Brian Mallari
# Objective: Set up a machine to detect faces in a set of pictures

# import modules here
import os
import json
import requests
import shutil

import cv2
import numpy as np

# Acquire pictures for training

# Create a directory to hold the pictures if such a file doesn't already exist
preDetectionDirectoryName = 'folderForPictures'
if not os.path.exists(preDetectionDirectoryName):
    os.makedirs(preDetectionDirectoryName)
    
    # Download the necessary pictures from the associated .json file for face detection
    pictureList = []
    print("Reading .json file which contains multiple JSON objects..." + "\n")
    with open('face_detection.json') as json_face_detection_file:
        for jsonObj in json_face_detection_file:
            facePictureDict = json.loads(jsonObj)
            pictureList.append(facePictureDict)

    print("Saving pictures from URL associated with each JSON decoded object (this may take several minutes)..." + "\n")
    currentDirectory = os.getcwd()

    # split the original directory into a list consisting of its hierarchical components
    splitDirectory = currentDirectory.split("\\") 
    splitDirectory.append(preDetectionDirectoryName) # add the directory for the face detection pictures to the list
    newSplitDirectory = splitDirectory # assign the modified list to a better-fitting variable name

    glue = "\\" # establish the "glue" for string join
    directoryForPictures = glue.join(newSplitDirectory)
 
    for picture in pictureList:
        url = picture['content']
        generalPictureName = "faceDetectionPicture_"
        filename = generalPictureName + str(pictureList.index(picture) + 1)
        fileExtension = url.split('.')[-1]
        fullFileName = filename + '.' + fileExtension
        r = requests.get(url, allow_redirects=True)
        open(fullFileName, 'wb').write(r.content)

        sourceDirectoryandFile = currentDirectory + "\\" + fullFileName
        destinationDirectoryandFile = directoryForPictures + "\\" + fullFileName
        
        shutil.move(sourceDirectoryandFile, destinationDirectoryandFile)
        
        if os.path.exists(fullFileName):
            os.remove(sourceDirectoryandFile)

    print("Pictures are now in the directory 'folderForPictures'" + "\n")

# set up face detection "module"

currentDirectory = os.getcwd()
prototxtFileName = "deploy.prototxt"
modelFileName = "res10_300x300_ssd_iter_140000.caffemodel"

# load serialized model from disk
print("Loading face detection model..." + "\n")
net = cv2.dnn.readNetFromCaffe(prototxtFileName, modelFileName)

# iterate over each file in the directory of face pictures
currentDirectory = os.getcwd()
splitDirectory = currentDirectory.split("\\")   # split the original directory into a list consisting of its hierarchical components
splitDirectory.append(preDetectionDirectoryName)  # add the directory for the face detection pictures to the list
newSplitDirectory = splitDirectory  # assign the modified list to a better-fitting variable name

glue = "\\" # establish the "glue" for string join
directoryForPictures = glue.join(newSplitDirectory)

print("Attempting to detect faces in each of the pictures in the directory 'folderForPictures' (this may take several minutes)..." + "\n")

directory = directoryForPictures
for entry in os.scandir(directory):
    if (entry.path.endswith(".jpg") or entry.path.endswith(".jpeg") or entry.path.endswith(".png")) and entry.is_file():
        entryPath = entry.path
              
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        image = cv2.imread(entryPath)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        # print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            defaultConfidence = 0.5 # this default value was taken from the source code 
            # featured on the webpage https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
            
            # if confidence > args["confidence"]:
            if confidence > defaultConfidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Create a directory to hold the post-detection pictures if such a file doesn't already exist
        postDetectionDirectoryName = 'folderForPictures_postDetection'
        if not os.path.exists(postDetectionDirectoryName):
            os.makedirs(postDetectionDirectoryName)
        
        # now try to save the imshow image...
        entryName = entry.name
        fileExtension = entryName.split('.')[-1]
        fileName = entryName.split('.')[-2]
        addendum = "_post_detection"
        newFullFileName = fileName + addendum + '.' + fileExtension
        cv2.imwrite(newFullFileName, image)

        # now try to move the saved image to the directory for post-detection pictures...
        currentDirectory = os.getcwd()

        # split the original directory into a list consisting of its hierarchical components
        splitDirectory = currentDirectory.split("\\") 
        splitDirectory.append(postDetectionDirectoryName) # add the directory for the face detection pictures to the list
        newSplitDirectory = splitDirectory # assign the modified list to a better-fitting variable name

        glue = "\\" # establish the "glue" for string join
        newDirectoryForPictures = glue.join(newSplitDirectory)

        sourceDirectoryandFile = currentDirectory + "\\" + newFullFileName
        postDetectionDestinationDirectoryandFile = newDirectoryForPictures + "\\" + newFullFileName
        
        shutil.move(sourceDirectoryandFile, postDetectionDestinationDirectoryandFile)
        
        if os.path.exists(newFullFileName):
            os.remove(sourceDirectoryandFile)
    
print("Post-detection pictures are now in the directory 'folderForPictures_postDetection'" + "\n")