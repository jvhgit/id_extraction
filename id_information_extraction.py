## Do not run on command line but as interactive session  ##
## 16 GB RAM, Intel i7 and RTX 2060 should do the trick ##
## This script will generate data based on a fake Dutch ID, tries to extract the information
##  and only serves for POC-ish purposes.
## This script contains several comments but is far from clean, I suggest one prints the outputs to get a better view on things.
## at last, it is best to just run it in parts, this way one understands it better 

#libraries
import pandas as pd
import numpy as np
import PIL
import json
import glob
import fitz
import io
from PIL import Image, ImageOps, ImageDraw
import gc
from copy import deepcopy

#load in original template image
image = Image.open('id_template.jpg');w, h = image.size

#resize for memory purposes
wpercent = (300/float(image.size[0]))
hsize = int((float(image.size[1])*float(wpercent)))
img = image.resize((300,hsize), Image.ANTIALIAS);img.size

#manually draw the boxes for the regions of interest (name and date of birth for now)
img_test = deepcopy(img)
draw=ImageDraw.Draw(img_test)
 
#name (make somewhat longer box since it is a short name)
draw.rectangle([(180,65),(107,53)],outline="red");img_test
#(180,65) - (x0, y0),(107,53) - (x1, y1)
# low-right: (x0, y0)
# upper-right: (x0, y1)
# low-left: (x1, y0)
# upper-left: (x1, y1)

#date of birth
draw.rectangle([(193,65 +48),(107,53+48)],outline="red");img_test
# (193,113) - (x0, y0),(107,101) - (x1, y1)
# low-right: (x0, y0)
# upper-right: (x0, y1)
# low-left: (x1, y0)
# upper-left: (x1, y1)

#convert image to array
image_num = np.asarray(img);image_num.shape
#convert back for extra check
# Image.fromarray(image_num)

#now generate samples with the template (moving the template around), i.e. add padding differently for each sample
def add_margin(pil_img, top, right, bottom, left, color):
    "From: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

#set image 'environment' size
width = 200; height = 200

#below is base position (since box coordinates are still valid now)
test_image = deepcopy(img)
im_new = add_margin(test_image, 0, width, height, 0, (255, 255, 255));im_new
draw=ImageDraw.Draw(im_new)
#name (make somewhat longer box since it is a short name)
draw.rectangle([(180,65),(107,53)],outline="red");im_new

def generate_data(n = 1000, with_noise = False):
    """
    Since there is no way to to gather real ID cards, I will generate different images using the template image.
    """
    num_samples = n
    list_images = []
    coordinates = []
    for _ in range(num_samples):
        #first generate two values from uniform dist (between 0 and 200)
        left_padding = np.random.randint(0, 200)
        top_padding = np.random.randint(0, 200)

        #copy original image 
        temp_image = deepcopy(img)

        if with_noise: #place image in random color
            temp_image = add_margin(
                            temp_image,
                            top_padding,
                            width - left_padding,
                            height - top_padding,
                            left_padding, 
                            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        )
        else: #place image in white frame
            temp_image = add_margin(
                            temp_image,
                            top_padding,
                            width - left_padding,
                            height - top_padding,
                            left_padding, 
                            (255, 255, 255)
                        )

        #now change also the box coordinates accordingly (name)
        x0, y0 = 180 + left_padding, 65 + top_padding
        x1, y1 = 107 + left_padding, 53 + top_padding 

        #(date of birth)
        x0_b, y0_b = 193 + left_padding, 65 + top_padding + 48
        x1_b, y1_b = 107 + left_padding, 53 + top_padding + 48

        #add to lists
        coordinates.append([ #these will be the regression targets
            x0, y0, x1, y1, 
            x0_b, y0_b, x1_b, y1_b
        ])
        list_images.append((np.asarray(temp_image)/255).astype('float16')) #standardize
        #delete and empty cache
        del temp_image; gc.collect()

    X = np.array(list_images).astype('float16') #memory issue so use less bits
    Y = np.array(coordinates).astype('float16') #memory issue so use less bits
    return X, Y

X,Y = generate_data( n = 6500, with_noise=True) #takes a long time
X_val, Y_val = generate_data(with_noise=True)
gc.collect()

#build first simple cnn model
if True: #these are my standard imports (most of them are probably redundant)
    import plotly.express as px
    import matplotlib.pyplot as plt
    from plotly import graph_objects as go
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    from tensorflow.keras.optimizers import SGD, Adam, Adamax
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, SpatialDropout1D, Flatten
    from tensorflow.keras.layers import (
        Dense,
        LSTM,
        Bidirectional,
        Dropout,
        SpatialDropout1D,
        Flatten,
        MaxPooling2D, 
        UpSampling2D
    )
    from tensorflow.keras import layers
    from tensorflow import keras
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np
    import os


model = Sequential()
model.add(layers.Conv2D(8, (5, 5), activation='relu', input_shape=(384, 500, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.GlobalMaxPooling2D())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(8))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(
                learning_rate=0.0005),
              loss = 'mse',
              metrics=['mse'])

history = model.fit(X, Y, epochs=25, batch_size=45, 
                    validation_data=(X_val, Y_val))
# This model is trained on the generated data containing only horizontal generated images.
gc.collect()
predictions = model.predict(X_val[:100])
i = 62
x0_pred, y0_pred, x1_pred, y1_pred, x0_pred_b, y0_pred_b, x1_pred_b, y1_pred_b =np.round(predictions[i])

val_image = Image.fromarray((X_val[i]*255).astype(np.uint8))
draw=ImageDraw.Draw(val_image)
draw.rectangle([(x0_pred,y0_pred),(x1_pred,y1_pred)],outline="red")#;temp_image
draw.rectangle([(x0_pred_b,y0_pred_b),(x1_pred_b,y1_pred_b)],outline="red");val_image

im1 = deepcopy(val_image).crop((x1_pred, y1_pred, x0_pred, y0_pred))
im2 = deepcopy(val_image).crop((x1_pred_b, y1_pred_b, x0_pred_b, y0_pred_b))

im1.show()
#(180,65) - (x0, y0),(107,53) - (x1, y1)
# low-right: (x0, y0)
# upper-right: (x0, y1)
# low-left: (x1, y0)
# upper-left: (x1, y1)

#now ocr the found boxes
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

custom_config = r'--oem 1 --psm 11'
pytesseract.image_to_string(im2, config=custom_config, lang='nld')

#we can also more variance in the data by rotating the image
#this requires to pin the coordinates again (transposing does not work directly)

img_test2 = deepcopy(img)
img_test2 = img_test2.transpose(Image.ROTATE_90);img_test2
draw2=ImageDraw.Draw(img_test2) 
#name (make somewhat longer box since it is a short name)
draw2.rectangle([(65,194),(53,121)],outline="red");img_test2
draw2.rectangle([(65 +48, 193),(53+48,107)],outline="red");img_test2

img2 = deepcopy(img)
img2 = img2.transpose(Image.ROTATE_90);img2 
def generate_data2(n = 1000, with_noise = False):
    "Basically the same as the other generating function but now the images is also set vertically"
    num_samples = n
    list_images = []
    coordinates = []
    #normal number of images
    for _ in range(int(num_samples/2)):
        #first generate two values from uniform dist (between 0 and 200)
        left_padding = np.random.randint(0, 200)
        top_padding = np.random.randint(0, 200)

        #copy original image 
        temp_image = deepcopy(img)

        #now use random values to place the image into white frame
        if with_noise:
            temp_image = add_margin(
                            temp_image,
                            top_padding,
                            width - left_padding,
                            height - top_padding,
                            left_padding, 
                            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        )#;temp_image.show()
        else:
            temp_image = add_margin(
                            temp_image,
                            top_padding,
                            width - left_padding,
                            height - top_padding,
                            left_padding, 
                            (255, 255, 255)
                        )

        #now change also the box coordinates accordingly (name)
        x0, y0 = 180 + left_padding, 65 + top_padding
        x1, y1 = 107 + left_padding, 53 + top_padding 

        #date of birth
        x0_b, y0_b = 193 + left_padding, 65 + top_padding + 48
        x1_b, y1_b = 107 + left_padding, 53 + top_padding + 48
        #add_coordinates
        # draw=ImageDraw.Draw(temp_image)
        # draw.rectangle([(x0,y0),(x1,y1)],outline="red")#;temp_image
        # draw.rectangle([(x0_b,y0_b),(x1_b,y1_b)],outline="red");temp_image

        #add to lists
        coordinates.append([
            x0, y0, x1, y1, 
            x0_b, y0_b, x1_b, y1_b
        ])
        list_images.append((np.asarray(temp_image)/255).astype('float16')) #standardize
        #delete and empty cache
        del temp_image; gc.collect()

    #rotated picture
    for _ in range(int(num_samples/2)):
        #first generate two values from uniform dist (between 0 and 200)
        left_padding = np.random.randint(-116, 200 )
        top_padding = np.random.randint(116,200)

        #copy original image 
        temp_image = deepcopy(img2)

        #now use random values to place the image into white frame
        if with_noise:
            temp_image = add_margin(
                            temp_image,
                            top_padding -116  ,
                            height - left_padding ,
                            width - top_padding  ,
                            left_padding + 116 , 
                            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        )#; temp_image.show()
        else:
            temp_image = add_margin(
                            temp_image,
                            top_padding -116,
                            width - left_padding + 116,
                            height - top_padding ,
                            left_padding + 116, 
                            (255, 255, 255)
                        )

        #now change also the box coordinates accordingly (name)
        # x0, y0 =  65 + top_padding -116, 194 + left_padding + 116
        # x1, y1 = 53 + top_padding - 116, 121 + left_padding + 116
        x0, y0 =  65 + left_padding + 116, 194 + top_padding - 116
        x1, y1 = 53 + left_padding + 116, 121 + top_padding - 116
        #date of birth
        # x0_b, y0_b =  65 + top_padding-116 + 48,193 + left_padding + 116
        # x1_b, y1_b =  53 + top_padding -116 + 48,107 + left_padding + 116
        
        x0_b, y0_b =  65 + left_padding+116 + 48,193 + top_padding - 116
        x1_b, y1_b =  53 + left_padding +116 + 48,107 + top_padding - 116
        
        #add_coordinates
        # draw=ImageDraw.Draw(temp_image)
        # draw.rectangle([(x0,y0),(x1,y1)],outline="red")#;temp_image
        # draw.rectangle([(x0_b,y0_b),(x1_b,y1_b)],outline="red");temp_image

        #add to lists
        coordinates.append([
            x0, y0, x1, y1, 
            x0_b, y0_b, x1_b, y1_b
        ])
        list_images.append((np.asarray(temp_image)/255).astype('float16')) #standardize
        #delete and empty cache
        del temp_image; gc.collect()
    # for vec in list_images: print(vec.shape)
    X_ = np.array(list_images).astype('float16')
    Y_ = np.array(coordinates).astype('float16')
    return X_, Y_

# del X_val, X, Y_val, Y; gc.collect()
X_rot,Y_rot = generate_data2( n = 7000, with_noise=True)
X_rot_val,Y_rot_val = generate_data2( n = 500, with_noise=True)

# X_val_rot, Y_val_rot = generate_data(n=4, with_noise=True)
k = 7
img_test3 = Image.fromarray((X_rot[k]*255).astype(np.uint8))
draw2=ImageDraw.Draw(img_test3)
#name (make somewhat longer box since it is a short name)
draw2.rectangle([(Y_rot[k][0],Y_rot[k][1]),(Y_rot[k][2],Y_rot[k][3])],outline="red");img_test3
draw2.rectangle([(Y_rot[k][4], Y_rot[k][5]),(Y_rot[k][6], Y_rot[k][7])],outline="red");img_test3



#built second cnn model (running into memory issues ) with a bigger model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(384, 500, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(8))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(
                learning_rate=0.0005)
              loss = 'mse',
              metrics=['mse'])
# gc.collect()
history = model.fit(X_rot, Y_rot, epochs=50, batch_size=45, 
                    validation_data=(X_rot_val, Y_rot_val))
# del model
predictions = model.predict(X_rot_val[100:400])
i = 272
x0_pred, y0_pred, x1_pred, y1_pred, x0_pred_b, y0_pred_b, x1_pred_b, y1_pred_b =np.round(predictions[i])


im1 = Image.fromarray((X_rot_val[i+100]*255)\
    .astype(np.uint8)).crop((x1_pred , y1_pred, x0_pred, y0_pred))
im2 = Image.fromarray((X_rot_val[i+100]*255)\
    .astype(np.uint8)).crop((x1_pred_b-2, y1_pred_b-2, x0_pred_b+2, y0_pred_b+2))

val_image = Image.fromarray((X_rot_val[i+100]*255).astype(np.uint8))
draw=ImageDraw.Draw(val_image)
draw.rectangle([(x0_pred,y0_pred),(x1_pred,y1_pred)],outline="red")#;temp_image
draw.rectangle([(x0_pred +1 ,y0_pred + 1),(x1_pred -1,y1_pred-1)],outline="red");val_image#;temp_image
draw.rectangle([(x0_pred_b +3,y0_pred_b+3),(x1_pred_b-3,y1_pred_b-3)],outline="red");val_image

np.asarray(im1).shape
if np.asarray(im1).shape[0] > np.asarray(im1).shape[1]:
    im1 = im1.transpose(Image.ROTATE_270)
    # print('Rotated')
    np.asarray(im1).shape

if np.asarray(im2).shape[0] > np.asarray(im2).shape[1]:
    im2 = im2.transpose(Image.ROTATE_270)
    # print('Rotated')


#now ocr the found boxes
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

custom_config = r'--oem 3 --psm 10' # good explanation of the config: https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc
val_image
text = pytesseract.image_to_string(im1 , config=custom_config, lang='nld') #dutch language, see for other languages (files for ocr engine): https://github.com/tesseract-ocr/tessdata 
text = text.replace('\n', '')
text = text.replace('\x0c', '');print(f'Name: {text}')
text = pytesseract.image_to_string(im2 , config=custom_config, lang='nld')
text = text.replace('\n', '')
text = text.replace('\x0c', '');print(f'Date of birth: {text}')
