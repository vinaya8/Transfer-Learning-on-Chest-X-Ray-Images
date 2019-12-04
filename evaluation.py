from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import pickle

fpr_vgg = pickle.load(open("vgg/fpr.pkl","rb"), encoding='latin1')
tpr_vgg = pickle.load(open("vgg/tpr.pkl","rb"), encoding='latin1')

fpr_mobile = pickle.load(open("mobilenet/fpr.pkl","rb"), encoding='latin1')
tpr_mobile = pickle.load(open("mobilenet/tpr.pkl","rb"), encoding='latin1')

fpr_inception = pickle.load(open("inception/fpr.pkl","rb"), encoding='latin1')
tpr_inception = pickle.load(open("inception/tpr.pkl","rb"), encoding='latin1')

fpr_densenet = pickle.load(open("densenet/fpr.pkl","rb"), encoding='latin1')
tpr_densenet = pickle.load(open("densenet/tpr.pkl","rb"), encoding='latin1')

plt.plot(fpr_vgg,tpr_vgg,label="VGG19")
plt.plot(fpr_mobile,tpr_mobile,label="MobileNet")
plt.plot(fpr_inception,tpr_inception,label="InceptionResNet")
plt.plot(fpr_densenet,tpr_densenet,label="DenseNet")

plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()