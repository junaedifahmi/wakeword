# Inference engine using tflite
This is the inference engine using to inference audio file using tensorflow lite model. 
To use this engine there is some requirement you should fulfill. See below for details.
This is not a script to train model, this scipt is just used to make prediction. 

# Install tflite library
To successfully run this engine you should first install tflite library. This can be obtain by follow this instruction from the [official tensorflow lite page](https://www.tensorflow.org/lite/guide/python). 
After succesfully install the package, please install the additional package that can be found in the `requirement` file in this repository.

# Convert model
After you train a model using tensorflow (not tensorflow lite), you will get a model either it is in **pb** format or **h5** format. You have to convert that model using `tflite_convert` binary that can be obtained when you install tensorflow package. Please refer [here](https://www.tensorflow.org/lite/convert/cmdline_examples) for detail.

# How to run
To run this scipt simply run with this sciprt
```sh
python inference_lite.py <model> <audio>
```
