# Transfer Learning on Chest X-Ray Images
Application of Transfer Learning on Chest X-Ray Images to detect Pneumonia

## Dataset
[Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification](https://data.mendeley.com/datasets/rscbjbr9sj/2)

Download the dataset and place it in the root of repository with name 'dataset' and containing the training, testing and validation dataset.

## Transfer Learning pre-trained models used
1. VGG19
2. InceptionResNet
3. MobileNet
4. DenseNet

## Implementation

#### Cloning the repository
```
git clone https://github.com/vinaya8/Transfer-Learning-on-Chest-X-Ray-Images.git
```
#### Installing the dependencies
```
pip install requirements.txt
```
#### Training the Model (Eg. VGG19)
```
cd vgg
python train.py
```

#### Testing the Model
```
python test.py
```

## Trained model weights
The trained weights for best customised models in all the pre-trained techniques can be downloaded from the following [link](https://www.dropbox.com/sh/bz2tmd1qojg3lnb/AAD67fhFUN32Gzz5pw4Pfc6aa?dl=0) 

## Custom Models Architecture
The custom models architecture made from the pre-trained models can be accessed in the JSON format from the following [link](https://www.dropbox.com/sh/hv37l5snpv9rxt4/AAB5h05b8D1d4I1LRyj9qRHTa?dl=0)