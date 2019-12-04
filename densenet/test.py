from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import pickle

test_dir = '../dataset/test/'
# load json and create model
json_file = open('../model_arch/densenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../models/densenet_best_model")

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
                                                 shuffle=False)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#loss, acc=model.evaluate_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

Y_pred = model.predict_generator(test_generator, STEP_SIZE_TEST)
y_pred = np.argmax(Y_pred, axis=1)

print("Accuracy:-")
print(accuracy_score(test_generator.classes, y_pred))

fpr, tpr, thresholds = roc_curve(test_generator.classes, y_pred)
roc_auc = auc(test_generator.classes, y_pred)

pickle.dump(fpr,open("fpr.pkl","wb"))
pickle.dump(tpr,open("tpr.pkl","wb"))

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
#print(acc)