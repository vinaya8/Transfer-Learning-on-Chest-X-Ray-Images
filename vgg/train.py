import os
from keras.layers import Dense, Conv2D, MaxPooling2D, AvgPool2D, Input, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import GlobalAveragePooling2D

train_dir = '../dataset/train/'
test_dir = '../dataset/test/'
val_dir = '../dataset/val/'

base_learning_rate = 1e-5
batch_size=32
epochs = 8

#Train Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

#Test Generator
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

#Validation Generator
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(train_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=30,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_generator=test_datagen.flow_from_directory(test_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=30,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_generator=val_datagen.flow_from_directory(val_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 class_mode='categorical',
                                                 shuffle=True)


base_model=VGG19( weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable=False

x=base_model.output
x=GlobalAveragePooling2D()(x)

x=Dense(2048,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)

preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[17:]:
    layer.trainable=True

def print_layers(model):
    for idx, layer in enumerate(model.layers):
        print("layer {}: {}, trainable: {}".format(idx, layer.name, layer.trainable))

print_layers(model)

adam = Adam(lr = base_learning_rate)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-7)

model_chkpoint = ModelCheckpoint(filepath='vgg_19_best_model', save_best_only=True, save_weights_only=True)

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size

model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=step_size_train, 
                    callbacks=[reduce_lr, model_chkpoint], validation_data=validation_generator,
                    validation_steps=step_size_val)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)