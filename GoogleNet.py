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
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], data_augmentation=imgaug, name='input')
conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
pool1_3_3 = local_response_normalization(pool1_3_3)
conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
conv2_3_3 = local_response_normalization(conv2_3_3)
pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

# 3a
inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3,  activation='relu', name='inception_3a_3_3')
inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

# 3b
inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name='inception_3b_5_5')
inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

# 4a
inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

# 4b
inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

# 4c
inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

# 4d
inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

# 4e
inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

# 5a
inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')

# 5b
inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3, activation='relu', name='inception_5b_3_3')
inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
pool5_7_7 = dropout(pool5_7_7, 0.4)

# fc
loss = fully_connected(pool5_7_7, 196, activation='softmax', weights_init = tflearn.initializations.xavier())
network = regression(loss, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name='targets')

# to train
model = tflearn.DNN(network, checkpoint_path='model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')

else:
    model.fit(X_train, y_train, n_epoch=650,validation_set=0.15,shuffle=True,
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
