import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras.layers.core import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import keras
from keras import models, layers, optimizers
from tensorflow.keras.layers import Input
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np
import pandas as pd
import cv2
import sklearn.metrics as metrics
# load the model

def define_model():
	resnet_model = Sequential()

	pretrained_model= tf.keras.applications.ResNet50(include_top=False,
	                   input_shape=(180,180,3),
	                   pooling='avg',classes=5,
	                   weights='imagenet')
	for layer in pretrained_model.layers:
	        layer.trainable=False

	resnet_model.add(pretrained_model)

	resnet_model.add(Flatten())
	resnet_model.add(Dense(512, activation='relu'))
	resnet_model.add(Dense(2, activation='softmax'))

	resnet_model.summary()

	resnet_model.compile(optimizer=Adam(lr=0.001),
						loss='categorical_crossentropy',
						metrics=['acc'])

	return resnet_model



# datagen creation 

def datagens():
    image_size = 224
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=20,
          width_shift_range=0.2,
          height_shift_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Change the batchsize according to your system RAM
    train_batchsize = 100
    val_batchsize = 10

    train_dir = "/home/s0r0eab/imagelist/images_ds/train"
    validation_dir = "/home/s0r0eab/imagelist/images_ds/validation"

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)
    
    return train_generator, validation_generator


def fit_model():
    model = define_model()
    train_generator, validation_generator = datagens()
    # Train the model
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_generator.samples/train_generator.batch_size ,
          epochs=10,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples/validation_generator.batch_size,
          verbose=1)

    # Save the model
    model.save('resnet_0809.h5')



fit_model()


def get_model_acc(saved_model_filename):
    model_saved = keras.models.load_model(saved_model_filename)

    test_datagen = ImageDataGenerator(rescale=1./255)
    pred_generator = test_datagen.flow_from_directory(
        directory='/home/s0r0eab/imagelist/images_ds/test',
        target_size=(224, 224),
        #color_mode="rgb",
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
    )

    acc = model_saved.evaluate(pred_generator)[1]
    #predictions = model_saved.predict_generator(pred_generator)
    print ("Accuracy is:", acc)
get_model_acc('resnet_0809.h5')
model_saved = keras.models.load_model('resnet_0809.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
pred_generator = test_datagen.flow_from_directory(
    directory='/home/s0r0eab/imagelist/images_ds/test',
    target_size=(224, 224),
    #color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)

y_prob=model_saved.predict_generator(pred_generator, verbose=1)
y_classes = y_prob.argmax(axis=-1)

label_lookup = {v: k for k, v in pred_generator.class_indices.items()}

count = 0

file_name = []
true_label = []
predicted_label = []

for path, pred in zip(pred_generator.filepaths, y_classes): 
    true_label.append(path.split('/')[-2])
    predicted_label.append(label_lookup[pred])
    file_name.append(path.split('/')[-1])
    #print('------------------------------------------------------\n\n')
    count +=1 



image_results = pd.DataFrame({"Filename":file_name,"True_Label":true_label,"Predicted_Label":predicted_label})
image_results.to_csv("/home/s0r0eab/imagelist/images_ds/output_resnet.csv")



Y_pred = model_saved.predict_generator(pred_generator, 373 // pred_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = metrics.confusion_matrix(pred_generator.classes, y_pred)
print(cm)
print('Classification Report')
print(metrics.classification_report(pred_generator.classes, y_pred))
