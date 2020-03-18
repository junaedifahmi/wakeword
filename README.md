# Wakeword
Wake word modeling using Tensorflow

## Prerequirements
Before running this code, please install a new virtual environtment with the requirements in the `requirement` file.

## How to run
After successfully install all the requirements, simply run this command in the console. Make sure that you activate your virtual environtments before hand.
```shell
python cnn.py <modelpath> <audiopath>
```
## Model Explanation
The model is using CNN Architecture with one fully connected layer. The accuracy gained from this model is a round 91% using blind test and 98% using training data. 
