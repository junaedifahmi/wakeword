# Wakeword
Wake word modeling using Tensorflow

## Prerequirements
Before running this code, please install a new virtual environtment with the requirements in the `requirement` file.

## How to run
After successfully install all the requirements, simply run this command in the console. Make sure that you activate your virtual environtments before hand.
```shell
python cnn.py <model> <audio>
```

## How this model is trained
### Feature extraction
The feature extraction is using MFCC with 13 coefficient. Because the data length vary from one to another, thus the extracted feature has different dimentions. Virous length of feature is not good for the model, so we overcome this problem by using padding to each feature, hence all the feature would have the same dimention. We pad all feature along the value and keep the number of coefficient the same (which is 13). The final dimention that pass the input layer is (13, 884), any feature that have bigger or smaller from that would be padded or trimemd.

### Training Hyperparameter
The training parameters in the model is left by default value in the `fit()` function of keras model. 


## Model Explanation
The model is using CNN Architecture with one fully connected layer. The accuracy gained from this model is a round 91% using blind test and 98% using training data. 
