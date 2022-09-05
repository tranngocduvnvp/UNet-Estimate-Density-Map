import numpy as np
from random import Random
import os
import pickle

import tensorflow as tf
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.backend import set_session
from keras.models import Sequential, Model, load_model

import matplotlib.pyplot as plt

# Some other specialist functions which could easily be substituted.
from PIL import Image  # for image import and output
from scipy import ndimage  # for Gaussian filtering
import cv2  # for padding function of image border.
from sklearn.model_selection import train_test_split

from model_2 import unet_model 
from generator import ImageDataGenerator, split_the_images  # additional functions


class TestCallback(Callback):
    """This function reports back the progress of the learning."""
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        self.losses = []

        
def step_decay(epoch):    
    """This sets up the various decays of the learning rate."""
    step = 16
    num = epoch // step
   
    if num > 2:
        lr = 1e-6
    else:
        if num % 3 == 0:
            lr = 1e-3
        elif num % 3 == 1:
            lr = 1e-4
        elif num % 3:
            lr = 1e-5
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def get_unet(img_rows, img_cols, channels=3):
    return unet_model(input_size=(img_rows, img_cols, channels))


def return_shuffled_im(train_size, test_size, input_seq):
    """returns shuffled indices for test and training data."""
    input_seq = input_seq.astype(np.int32)
    image_pool_size = np.array(input_seq).shape
    rand_seq = np.random.choice(input_seq, input_seq.shape[0], replace=True)

    print(train_size, test_size, image_pool_size, sep='\t')
    # assert (train_size + test_size) <= image_pool_size,\
    # 'your combined test and train images exceeds the number of images in your input sequence.'

    shuf_train_images = rand_seq[:train_size]
    shuf_test_images = rand_seq[train_size:train_size + test_size]

    return shuf_train_images, shuf_test_images


"""## Training Model"""

#generate how many models.
models_to_gen = 5

for i in range(models_to_gen):

    file_path = 'dataset02/' # Dataset to use.
    data_store = {}
    data_store['input'] = []
    data_store['gt'] = []
    data_store['dense'] = []

    #Parameters of fit.
    in_hei = 96 #Size of each image patch
    in_wid = 96 #Size of each image patch
    mag = 16 #The padding margin to use
    train_size = 72 #The number of training images.
    test_size = 8 #The number of test images.
    learning_rate = "1e-5 see learning rate decay"
    batch_size = 16 #For stochastic gradient descent, the number of images in each batch.
    nb_epoch = 50 #The number of epochs
    samples_per_epoch = 60 #For stochastic gradient descent, the number of batches per epoch.
    image_pool_size = train_size+test_size
    input_seq = np.arange(0,image_pool_size)
    sigma = 2.0 #Size of kernel representing the ground-truth density.
    save_best_only = False
    loss = 'mse' # I use mean square error in this case.

    decay = 0.0


    # Experiment number should match filename
    filepath = "saved_models/keras_tf_exp_sdataset02s2_model_" + str(i)

    for i in range(0,image_pool_size):
        n = str(i+1).zfill(3)

        #Open intensity image.
        img = Image.open(file_path + n + 'cells.png').getdata()
        wid, hei = img.size
        temp = np.array(img).reshape((hei,wid,3))[:,:,2].astype(np.float32)

        data_store['input'].append(temp)

        #Open ground-truth image.
        img =  Image.open(file_path + n + 'dots.png').getdata()
        data_store['gt'].append(np.array(img).reshape((hei,wid))[:,:].astype(np.float64))

        #Filter ground-truth image to produce density kernel representation
        data_store['dense'].append(ndimage.filters.gaussian_filter(data_store['gt'][i],sigma,mode='constant'))


    train = []
    gtdata = []

    shuf_train_images, shuf_test_images = return_shuffled_im(train_size,test_size,input_seq)

    X_trainf = []
    Y_trainf = []
    X_testf = []
    Y_testf = []

    for i in shuf_train_images:
        X_trainf.append(data_store['input'][i])
        Y_trainf.append(data_store['dense'][i])
    for i in shuf_test_images:
        X_testf.append(data_store['input'][i])
        Y_testf.append(data_store['dense'][i])

    # X_trainf, X_testf, Y_trainf, Y_testf = train_test_split(data_store['input'],  data_store['dense'], train_size = train_size,test_size=18)
    train_cut, train_gtdata_cut, images_per_image = split_the_images(X_trainf, Y_trainf, in_hei, in_wid,mag)
    test_cut, test_gtdata_cut, images_per_image = split_the_images(X_testf, Y_testf,in_hei,in_wid,mag)


    X_train = np.array(train_cut)
    Y_train = np.array(train_gtdata_cut)  
    X_test = np.array(test_cut)
    Y_test = np.array(test_gtdata_cut)


    #Stores history of paramaters.
    hist = keras.callbacks.History()


    # combine generators into one which yields image and masks
    #train_generator = zip(image_generator, mask_generator)
    height = X_train.shape[2]
    width = X_train.shape[3]
    model = get_unet(height, width, 1)

    X_train = np.swapaxes(X_train,1,3)
    X_train = np.swapaxes(X_train,2,1)
    Y_train = np.swapaxes(Y_train,1,3)
    Y_train = np.swapaxes(Y_train,2,1)
    X_test = np.swapaxes(X_test,1,3)
    X_test = np.swapaxes(X_test,2,1)
    Y_test = np.swapaxes(Y_test,1,3)
    Y_test = np.swapaxes(Y_test,2,1)


    print('fitting model ', X_train.shape, np.max(X_train))
    checkpoint = ModelCheckpoint(filepath + ".hdf5",  monitor='loss', verbose=1, save_best_only=save_best_only, mode='min')



    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
        zoom_range = 0.3,
        shear_range = 0.,
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True,
        fill_mode = 'constant',
        dim_ordering = 'tf')  # randomly flip images


    change_lr = LearningRateScheduler(step_decay)

    callbacks_list = [checkpoint, change_lr]



    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), steps_per_epoch  = samples_per_epoch, epochs=nb_epoch, callbacks=callbacks_list, verbose=True, validation_data=(X_test, Y_test))

    #Store the paramters of the fit, along with the model and data preprocessing.
    hist.history['parameters'] = {}
    hist.history['parameters']['batch_size'] = batch_size
    hist.history['parameters']['nb_epoch'] = nb_epoch
    hist.history['parameters']['in_hei'] = in_hei
    hist.history['parameters']['in_wid'] = in_wid
    hist.history['parameters']['mag'] = mag
    hist.history['parameters']['sigma'] = sigma
    hist.history['parameters']['train_size'] = train_size
    hist.history['parameters']['learning_rate'] = learning_rate
    hist.history['parameters']['shuf_train_images'] = shuf_train_images
    hist.history['parameters']['shuf_test_images'] = shuf_test_images
    hist.history['parameters']['loss'] = loss
    hist.history['parameters']['save_best_only'] = save_best_only
    pickle.dump(hist.history, open(filepath + '.his', "wb"))


"""## Making predictions"""

## Load in the paramters.

model_path ='saved_models'
experiment = 'keras_tf_exp_sdataset02s2_model_'

imported_pickle = pickle.load(open(model_path + '/' + experiment + '0.his', "rb"))

parameters = imported_pickle['parameters']
file_path = 'dataset02/'
data_store = {}
data_store['input'] = []
data_store['gt'] = []
data_store['dense'] = []
train = []
gtdata = []
in_hei = parameters['in_hei']
in_wid = parameters['in_wid']
mag = parameters['mag']
learning_rate = parameters['learning_rate']
loss = parameters['loss']
save_best_only = parameters['save_best_only']


loss = 'mse'
sigma = parameters['sigma']
num_of_train = 19
test_size = 18


## Load the images for testing.
for i in range(num_of_train):
    n = str(i + 80).zfill(3)

    # Open intensity image.
    img = Image.open(file_path + n + 'cells.png').getdata()
    wid, hei = img.size
    temp = np.array(img).reshape((hei, wid, 3))[:, :, 2].astype(np.float32)

    data_store['input'].append(temp)

    # Open ground-truth image.
    img = Image.open(file_path + n + 'dots.png').getdata()
    data_store['gt'].append(np.array(img).reshape((hei, wid))[:, :].astype(np.float64))

    # Filter ground-truth image to produce density kernel representation
    data_store['dense'].append(ndimage.filters.gaussian_filter(data_store['gt'][i], sigma, mode='constant'))


X_trainf, X_testf, Y_trainf, Y_testf = train_test_split(data_store['input'],  data_store['dense'], train_size =1, test_size=test_size)

test_cut, test_gtdata_cut, images_per_image = split_the_images(X_testf, Y_testf, in_hei, in_wid, mag)

X_test = np.array(test_cut)
Y_test = np.array(test_gtdata_cut)

X_test = np.swapaxes(X_test, 1, 3)
X_test = np.swapaxes(X_test, 2, 1)

for vt in range(1):
    filename = model_path + "/" + experiment + "" + str(vt) + ".hdf5"
    # load json and create model

    # load weights into new model
    loaded_model = load_model(filename)
    print("Loaded model from disk")
    imported_pickle = pickle.load(open(model_path + '/' + experiment + "" + str(vt) + '.his', "rb"))

    f = open(model_path + '/' + experiment + str(vt) + 'out.txt', 'w+')

    loss_print = imported_pickle['loss']
    val_loss_print = imported_pickle['val_loss']
    f.write('Loss_pt\t\tVal_loss_pt\n')
    for loss_pt, val_pt in zip(loss_print, val_loss_print):
        f.write(str(loss_pt) + '\t\t' + str(val_pt) + '\n')
    f.close()

    # evaluate loaded model on test data
    # loaded_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    imgs_mask_test = loaded_model.predict(X_test, verbose=1)

    # Switches format for tensor flow.
    imgs_mask_test = np.swapaxes(imgs_mask_test, 2, 1)
    imgs_mask_test = np.swapaxes(imgs_mask_test, 1, 3)
    data_store['output'] = []
    c = 0

    null, f_hei, f_wid = imgs_mask_test[0].shape

    # reconstruct the images.
    while c < (imgs_mask_test.__len__() - images_per_image) + 1:
        out = np.zeros((hei, wid))
        rows = 0
        for rst in range(0, hei, in_hei):
            rows += 1
            cols = 0
            d = np.floor(float(c) / float(images_per_image))
            for cst in range(0, wid, in_wid):
                cols += 1
                img = imgs_mask_test[c]
                c += 1
                top = bottom = left = right = 16
                ren = rst + in_hei
                cen = cst + in_wid
                if ren > hei:
                    top = top + (ren-hei)
                    ren = hei
                if cen > wid:
                    right = right + (cen-wid)
                    cen = wid

                out[rst:ren, cst:cen] = img[0, bottom:-top, left:-right]

        data_store['output'].append(out)


    # Output of performance metrics
    pef = []
    trk = []
    abs_err = []
    inpu_arr = []
    pred_arr = []
    perc_arr = []

    print('Len data store: ', data_store['output'].__len__())
    print('Test size: ', test_size)
    # figsize(12,12)
    for idx in range(test_size):
        # figure()
        inpu_img = Y_testf[idx]
        pred_img = data_store['output'][idx]
        n = str(idx)
        b = n.zfill(3)

        pickle.dump(inpu_img, open("out_imgs/inpu_img" + b + ".p", "wb"))
        pickle.dump(pred_img, open("out_imgs/pred_img" + b + ".p", "wb"))

        inpu_sum = np.sum(inpu_img) / 255.0
        pred_sum = np.sum(pred_img) / 255.0

        inpu_arr.append(inpu_sum)
        pred_arr.append(pred_sum)
        abs_err.append(abs(inpu_sum-pred_sum))
        perc_arr.append((1-(abs(inpu_sum-pred_sum)/pred_sum))*100)

    print('Average Abs error: ', np.average(abs_err))
    print('Average input arr: ', np.average(inpu_arr))
    print('average pred: ', np.average(pred_arr))
    print('average perc: ', np.average(perc_arr))
    imported_pickle['pred_arr'] = pred_arr
    imported_pickle['perc_arr'] = perc_arr
    imported_pickle['abs_err'] = pred_arr
    imported_pickle['inpu_arr'] = perc_arr

    pickle.dump(imported_pickle, open(model_path + '/' + experiment + "" + str(vt) + '.his', "wb"))


# Visualization
for img_name in [i for i in os.listdir("out_imgs") if (i.split('.')[-1] != 'jpg' and i[:4] == 'inpu')]:
    ori_image = pickle.load(open(os.path.join("out_imgs", img_name), 'rb')).astype('uint8')
    _, thresh = cv2.threshold(ori_image, 1, 255, cv2.THRESH_BINARY)
    ori_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pre_image = pickle.load(open(os.path.join("out_imgs", "pred" + img_name[4:]), 'rb')).astype('uint8')
    _, thresh = cv2.threshold(pre_image, 1, 255, cv2.THRESH_BINARY)
    pre_contours, _i = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plt.figure(figsize=(20, 10))
    plt.suptitle(img_name + "Actual: " + str(len(ori_contours)) + ", Prediction: " + str(len(pre_contours)))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(ori_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Prediction Image")
    plt.imshow(pre_image, cmap='gray')
    plt.show()
