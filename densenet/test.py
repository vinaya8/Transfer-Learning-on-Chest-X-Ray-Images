from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.optimizers import Adam
import pandas as pd

test_dir = '../dataset/test/'
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("densenet_best_model")

base_learning_rate = 1e-5
adam = Adam(lr = base_learning_rate)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)
#print(loaded_model.summary())

#Test Generator
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator=test_datagen.flow_from_directory(test_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=30,
                                                 class_mode='categorical',
                                                 shuffle=True)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#test_generator.reset()
loss, acc=model.evaluate_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

print(acc)