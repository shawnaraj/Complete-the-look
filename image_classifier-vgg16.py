from keras.applications.vgg16 import VGG16
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
#%matplotlib inline
#from PIL import ImageFile
from IPython.display import Image, display
#ImageFile.LOAD_TRUNCATED_IMAGES = False
import sys
from tensorflow.keras.layers import Input
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np
import pandas as pd
import cv2
import sklearn.metrics as metrics
#%matplotlib inline
#from PIL import ImageFile
from IPython.display import Image, display
#ImageFile.LOAD_TRUNCATED_IMAGES = False
import sys
print(sys.version)
def define_model():
    
    #Load the VGG model
    image_size = 224
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    print(model.summary())
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    
    return model


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
    model.save('image_classifier_0805.h5')

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
get_model_acc('image_classifier_0805.h5')
model_saved = keras.models.load_model('image_classifier_0805.h5')

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
image_results.to_csv("/home/s0r0eab/imagelist/images_ds/output.csv")



Y_pred = model_saved.predict_generator(pred_generator, 373 // pred_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = metrics.confusion_matrix(pred_generator.classes, y_pred)
print(cm)
print('Classification Report')
print(metrics.classification_report(pred_generator.classes, y_pred))
