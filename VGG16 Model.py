import os
import joblib
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import tflearn
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy


#############################################################################

TRAIN_DIR = '../input/train/train'
TEST_DIR = '../input/test/test'
MODEL_NAME = 'Car_Type_Classification'
ANNO_TRAIN_DIR = '../input/anno_train.csv'
ANNO_TEST_DIR = '../input/anno_test.csv'
NAMES = '../input/names.csv'


train_dir_list = os.listdir(TRAIN_DIR)
anno_train = pd.read_csv(ANNO_TRAIN_DIR,header=None)
anno_test = pd.read_csv(ANNO_TEST_DIR,header=None)
names = pd.read_csv(NAMES,header=None)
anno_train[5]-=1


#############################################################################

def set_train_label(img_label):
    label_encoded = np.zeros(196)
    label_encoded[img_label] = 1
    return label_encoded

IMG_SIZE = 227
def LoadSave_train_data():
    training_data = []
    countH = 0
    for sub_dir in os.listdir(TRAIN_DIR):
        fldr = os.path.join(TRAIN_DIR,sub_dir)
        for img_name in tqdm(os.listdir(fldr)):
            img_index = int(img_name.split('.')[-2])-1 #extrating image index to use it in the dataframe
            path = os.path.join(fldr,img_name) # generating image path
            img_data = cv2.imread(path,0) # read the image in Gray mode
            img_data = np.divide(img_data,255) # normalize image to range [0-1]
            minx, miny, maxx, maxy = anno_train.iloc[img_index,1], anno_train.iloc[img_index,2], anno_train.iloc[img_index,3], anno_train.iloc[img_index,4]
            img_data = img_data[miny:maxy,minx:maxx] # crop the image
            img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE)) # resize the image (to use it later as input in input layer)
            vertical_img = img_data.copy()
            vertical_img = cv2.flip( img_data, 1 ) # flipping image
            blurred = img_data.copy()
            blurred = cv2.blur(img_data,(7,7)) # add random noise
            training_data.append([np.array(img_data),set_train_label(anno_train.iloc[countH,5])]) # append image with its label to the training list
            training_data.append([np.array(vertical_img),set_train_label(anno_train.iloc[countH,5])])
            training_data.append([np.array(blurred),set_train_label(anno_train.iloc[countH,5])])
            countH+=1
    shuffle(training_data)
    #np.save('train_data.npy', training_data)
    #joblib.dump(training_data,'train_data')
    print ('train data images readed Successfuly!')
    return training_data

def load_train_data():
    #training_data = np.load('train_data.npy')
    training_data = joblib.load('train_data',mmap_mode='r')
    return training_data


#############################################################################

train_data = None
print ('Start loading train data...')
train_data = LoadSave_train_data()

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print ('Done1')
y_train = [i[1] for i in train_data]
print ('Done2')


#############################################################################


tf.reset_default_graph()
imgaug = tflearn.ImageAugmentation()
imgaug.add_random_flip_leftright()
imgaug.add_random_rotation()
network = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], data_augmentation=imgaug, name='input',)

network = tflearn.conv_2d(network, 64, 3, activation='relu')
network = tflearn.conv_2d(network, 64, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2, name='maxpool1')

network = tflearn.conv_2d(network, 128, 3, activation='relu')
network = tflearn.conv_2d(network, 128, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2, name='maxpool2')

network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2, name='maxpool3')

network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2, name='maxpool4')

network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2, name='maxpool5')

network = tflearn.fully_connected(network, 4096, activation='relu')
network = tflearn.dropout(network, 0.5, name='dropout1')

network = tflearn.fully_connected(network, 4096, activation='relu')
network = tflearn.dropout(network, 0.5, name='dropout2')

cnn_layers = tflearn.fully_connected(network, 196, activation='softmax', weights_init = tflearn.initializations.xavier())
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')

else:
    model.fit(X_train, y_train, n_epoch=450,validation_set=0.15,shuffle=True,
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')


#############################################################################

ids = []
def LoadSave_test_data():
    testing_data = []
    countH = 0
    for sub_dir in os.listdir(TEST_DIR):
        fldr = os.path.join(TEST_DIR,sub_dir)
        for img_name in tqdm(os.listdir(fldr)):
            ids.append(img_name)
            img_index = int(img_name.split('.')[-2])-1 #extrating image index to use it in the dataframe
            path = os.path.join(fldr,img_name) # generating image path
            img_data = cv2.imread(path,0) # read the image in Gray mode
            img_data = np.divide(img_data,255) # normalize image to range [0-1]
            minx, miny, maxx, maxy = anno_test.iloc[img_index,1], anno_test.iloc[img_index,2], anno_test.iloc[img_index,3], anno_test.iloc[img_index,4]
            img_data = img_data[miny:maxy,minx:maxx] # crop the image
            img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE)) # resize the image (to use it later as input in input layer)
            testing_data.append(np.array(img_data)) # append image with its label to the training list
            countH+=1
    #np.save('train_data.npy', training_data)
    print ('test data images readed Successfuly!')
    return testing_data


#############################################################################

test_data = None
print ('Start loading test data...')
test_data = LoadSave_test_data()

X_test = np.array([img for img in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print ('Done1')



#############################################################################

total_predictions = []
startH, endH = 0, 100
for i in range(80):
    print (i)
    preds = model.predict(X_test[startH:endH])
    for j in range(100):
        MaxIndex = np.argmax(preds[j])
        total_predictions.append(MaxIndex+1)
    startH = endH
    endH+=100


for data_instance in X_test[8000:]:
    pred = model.predict([data_instance])[0]
    MaxIndex = np.argmax(pred)
    total_predictions.append(MaxIndex+1)

print ('predictions Done')
dataf = np.hstack((np.array(ids)[:,np.newaxis],np.array(total_predictions)[:,np.newaxis]))
print ('start data frame')
all_together = pd.DataFrame(data=dataf,columns=['id','label'])
all_together.to_csv('submit.csv',index=False)
print ('Done')
