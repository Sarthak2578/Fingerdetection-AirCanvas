#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install torch


# In[ ]:


pip install torchvision


# In[1]:


import cv2
import numpy as np 
import copy
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import PIL
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


rect1_tl = (620,240)
rect2_tl = (620,290)
rect3_tl = (620,340)
rect4_tl = (580,320)
rect5_tl = (660,320)

height = 30
width = 30

# Shift all the rectangles to the left by reducing the 'x' coordinate
shift_amount_x = 350
shift_amount_y = 70 # You can adjust this value based on your needs
rect1_tl = (rect1_tl[0] - shift_amount_x, rect1_tl[1] - shift_amount_y)
rect2_tl = (rect2_tl[0] - shift_amount_x, rect2_tl[1] - shift_amount_y)
rect3_tl = (rect3_tl[0] - shift_amount_x, rect3_tl[1] - shift_amount_y)
rect4_tl = (rect4_tl[0] - shift_amount_x, rect4_tl[1] - shift_amount_y)
rect5_tl = (rect5_tl[0] - shift_amount_x, rect5_tl[1] - shift_amount_y)


# In[3]:


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16,  3,padding = 1 )
        #Convultion Sees 28*28*16
        
        self.conv2 = nn.Conv2d(16, 32, 3,padding = 1)
        #convultion sees 14*14*32
        
        self.conv3 = nn.Conv2d(32, 64, 3,padding = 1)
        #convultion sees 7*7*64
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*3*3,256)
        self.fc2 = nn.Linear(256,26)
        self.dropout = nn.Dropout(0.25)
        # max pooling layer

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64*3*3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# In[4]:


def getHistogram(frame):
    roi1 = frame[rect1_tl[1]:rect1_tl[1]+width,rect1_tl[0]:rect1_tl[0]+height]
    roi2 = frame[rect2_tl[1]:rect2_tl[1]+width,rect2_tl[0]:rect2_tl[0]+height]
    roi3 = frame[rect3_tl[1]:rect3_tl[1]+width,rect3_tl[0]:rect3_tl[0]+height]
    roi4 = frame[rect4_tl[1]:rect4_tl[1]+width,rect4_tl[0]:rect4_tl[0]+height]
    roi5 = frame[rect5_tl[1]:rect5_tl[1]+width,rect5_tl[0]:rect5_tl[0]+height]
    roi = np.concatenate((roi1,roi2,roi3,roi4,roi5),axis = 0)
    roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    return cv2.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])


# In[5]:


def drawRectangles(frame = 0):
    frame_with_rect = frame
    cv2.rectangle(frame_with_rect,rect1_tl,tuple(np.array(rect1_tl)+np.array((height,width))),(0,0,255),1)
    cv2.rectangle(frame_with_rect,rect2_tl,tuple(np.array(rect2_tl)+np.array((height,width))),(0,0,255),1)
    cv2.rectangle(frame_with_rect,rect3_tl,tuple(np.array(rect3_tl)+np.array((height,width))),(0,0,255),1)
    cv2.rectangle(frame_with_rect,rect4_tl,tuple(np.array(rect4_tl)+np.array((height,width))),(0,0,255),1)
    cv2.rectangle(frame_with_rect,rect5_tl,tuple(np.array(rect5_tl)+np.array((height,width))),(0,0,255),1)
    return frame_with_rect


# In[6]:


def getMask(frame, histogram):
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([frame_hsv],[0,1],histogram,[0,180,0,256],1)
    _,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask = cv2.filter2D(mask,-1,kernel)

    kernel1 = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.bilateralFilter(mask,5,75,75)

    return mask


# In[7]:


def getMaxContour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    return max_contour


# In[8]:


def drawDefects(frame_with_rect,maxContour,hull):
    defects = cv2.convexityDefects(maxContour,hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(maxContour[s][0])
        end = tuple(maxContour[e][0])
        far = tuple(maxContour[f][0])
        cv2.line(frame_with_rect,start,far,[255,0,0],2)
        cv2.line(frame_with_rect,far,end,[0,255,0],2)
        cv2.circle(frame_with_rect,far,5,[0,0,255],-1)


# In[9]:


def getCentroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy


# In[10]:


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=float)
        y = np.array(contour[s][:, 0][:, 1], dtype=float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


# In[11]:


def cropCharacter(canvas):
    print(canvas.shape)
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            print('i {i}',i)
            print('j {j}',j)

            if canvas[i,j]!=255:
                canvas = canvas[i:canvas.shape[0],:]

    return canvas


# In[12]:


def getROI(canvas):
    # Invert the canvas to make the characters white on a black background.
    gray = cv2.bitwise_not(canvas)
    
    # Apply a binary threshold to obtain a binary image with characters as white.
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image.
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store areas and their corresponding contour indices.
    areas = []

    # Calculate the area of each contour's bounding box.
    for i in range(len(ctrs)):
        x, y, w, h = cv2.boundingRect(ctrs[i])
        areas.append((w * h, i))

    # Sort the areas in descending order.
    areas.sort(reverse=True)
    
    if areas:
        # Extract the bounding box of the largest area (first in the sorted list).
        x, y, w, h = cv2.boundingRect(ctrs[areas[0][1]])
        
        # Draw a rectangle around the detected ROI on the canvas.
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
        # Extract the ROI from the inverted grayscale image.
        roi = gray[y:y + h, x:x + w]
        
        # Return the ROI.
        return roi
    else:
        # If no areas are found, return None.
        return None


# In[13]:


def predictCharacter(roi, model):
    img = cv2.resize(roi, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = Image.fromarray(img)

    # Convert the grayscale image to a single channel (1x28x28)
    img = img.convert("L")

    normalize = transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    p_img = preprocess(img)

    model.eval()
    p_img = p_img.unsqueeze(0)  # Add a batch dimension
    output = model(p_img)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds


# In[14]:



def main():
    cap = cv2.VideoCapture(0)

    canvas = np.zeros((720,1280), np.uint8)

    far_points = []

    pressed = False
    isDrawing = False
    madePrediction = False
    # create a complete CNN
    model = Net()

    model.load_state_dict(torch.load('D:\Downloads\model_emnist.pt', map_location='cpu'))
    while True:
        _ , frame = cap.read()
        frame = cv2.flip(frame,flipCode = 1)
        originalFrame = copy.deepcopy(frame)
        originalFrame = drawRectangles(originalFrame)
        canvas[:,:] = 255

        key = cv2.waitKey(1)

        if key & 0xFF == ord('a'):
            pressed = True
            histogram = getHistogram(frame)
            
        # Inside the loop, after capturing the histogram:
        if pressed:
        # Normalize the histogram for visualization
            hist_normalized = cv2.normalize(histogram, None, 0, 255, cv2.NORM_MINMAX)

        # Create an image to represent the histogram
            hist_image = np.zeros((100, 256, 3), dtype=np.uint8)

        # Draw the histogram on the image
            for i in range(1, 256):
                x1 = i - 1
                y1 = 100 - int(hist_normalized[i - 1])
                x2 = i
                y2 = 100 - int(hist_normalized[i])
                cv2.line(hist_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Display the histogram image
            cv2.imshow('Histogram', hist_image)


            
        if key & 0xFF == ord('d'):
            isDrawing = True

        if key & 0xFF == ord('c'):
            canvas[:,:] = 255
            isDrawing = False
            far_points.clear()
            madePrediction = False


        if isDrawing:
            if len(far_points)>100:
                far_points.pop(0)
            far_points.append(far)
            for i in range(len(far_points)-1):
                cv2.line(originalFrame, far_points[i], far_points[i+1], (255,5,255), 10)
                cv2.line(canvas, far_points[i], far_points[i+1], (0,0,0), 10)

        if key & 0xFF == ord('f'):
            isDrawing = False
            #canvas = cropCharacter(canvas)
            #cv2.imshow('LeftCrop', canvas)
            roi = getROI(canvas)
            prediction = predictCharacter(roi, model)
            print(prediction)
            madePrediction = True
            name = str(prediction) + '.jpg'
            cv2.imwrite(name, roi)


        if pressed:
            mask = getMask(frame,histogram)
            maxContour = getMaxContour(mask)
            epsilon = 0.25*cv2.arcLength(maxContour,True) 
            approx = cv2.approxPolyDP(maxContour,epsilon,True)
            hull = cv2.convexHull(maxContour,returnPoints = False)
            drawDefects(originalFrame, maxContour, hull)
            defects = cv2.convexityDefects(maxContour,hull)
            far = farthest_point(defects, maxContour, getCentroid(maxContour))
            cv2.circle(originalFrame,far,10,[0,200,255],-1)


            #cv2.imshow('Mask', mask)
            #cv2.imshow('Canvas', canvas)

        if madePrediction:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'You wrote - ' + chr(prediction + 65)  # Include the letter you predicted
            # Adjust the font scale to change the font size
            font_scale = 2  # Change this value to adjust the font size

            text_size = cv2.getTextSize(text, font, font_scale, 2)[0]  # Get the size of the text

    # Adjust the position based on text size to keep it within the frame
            text_x = max(10, (originalFrame.shape[1] - text_size[0]) // 2)  # Centered horizontally
            text_y = max(100, originalFrame.shape[0] - 10)  # Place it near the bottom of the frame

            cv2.putText(originalFrame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        if key & 0xFF == ord('q'):
            break
        cv2.imshow('frame',originalFrame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# In[ ]:


'''
# Define the coordinates and size of the fixed drawing area (square)
draw_area_x = 100
draw_area_y = 100
draw_area_size = 200  # Adjust the size as needed

# ... (Previous code for defining the drawing area)

def main():
    cap = cv2.VideoCapture(0)

    canvas = np.zeros((720, 1280), np.uint8)

    far_points = []

    pressed = False
    isDrawing = False
    madePrediction = False
    model = Net()

    model.load_state_dict(torch.load('D:\Downloads\model_emnist.pt', map_location='cpu'))
    
    far = None  # Initialize far outside the isDrawing block

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, flipCode=1)
    originalFrame = copy.deepcopy(frame)
    originalFrame = drawRectangles(originalFrame)
    canvas[:,:] = 255

    key = cv2.waitKey(1)

    # Draw the fixed drawing area as a square
    cv2.rectangle(originalFrame, (draw_area_x, draw_area_y),
                  (draw_area_x + draw_area_size, draw_area_y + draw_area_size), (0, 0, 0), 2)

    if key & 0xFF == ord('d'):
        # Start drawing mode when 'd' key is pressed
        isDrawing = True
        far_points.clear()  # Clear the previous drawing points

    if key & 0xFF == ord('c'):
        # Clear the entire canvas (including the fixed drawing area)
        canvas[:,:] = 255
        madePrediction = False

    if key & 0xFF == ord('f'):
        # End drawing mode when 'f' key is pressed
        isDrawing = False
        pressed = True  # Indicate that drawing is completed and ready for prediction

    if isDrawing:
        if len(far_points) > 100:
            far_points.pop(0)
        far_points.append(far)
        for i in range(len(far_points) - 1):
            cv2.line(originalFrame, far_points[i], far_points[i + 1], (255, 5, 255), 10)
            cv2.line(canvas, far_points[i], far_points[i + 1], (0, 0, 0), 10)

    if pressed:
        mask = getMask(frame, histogram)
        maxContour = getMaxContour(mask)
        epsilon = 0.25 * cv2.arcLength(maxContour, True)
        approx = cv2.approxPolyDP(maxContour, epsilon, True)
        hull = cv2.convexHull(maxContour, returnPoints=False)
        drawDefects(originalFrame, maxContour, hull)
        defects = cv2.convexityDefects(maxContour, hull)
        far = farthest_point(defects, maxContour, getCentroid(maxContour))
        cv2.circle(originalFrame, far, 10, [0, 200, 255], -1)

    if madePrediction:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'You wrote - ' + chr(prediction + 65)  # Include the letter you predicted
        font_scale = 2  # Adjust the font size
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_x = max(10, (originalFrame.shape[1] - text_size[0]) // 2)
        text_y = max(100, originalFrame.shape[0] - 10)

        cv2.putText(originalFrame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    if key & 0xFF == ord('q'):
        break
    cv2.imshow('frame', originalFrame)


# In[ ]:




