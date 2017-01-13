# Let there be Color!: Automatic Colorization of Grayscale Images
![Teaser Image](https://raw.githubusercontent.com/satoshiiizuka/siggraph2016_colorization/master/example_results.png)

## Overview
This code provides an implementation of the [research paper](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf) in python using Tensorflow:
```
  "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification"
  Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa
  ACM Transaction on Graphics (Proc. of SIGGRAPH 2016), 2016
```

This paper provides a method of automatically coloring a grayscale images with a deep network. The network learns both local features and global features jointly in a single framework which can then be used on images of any resolution. By incorporating global features it is able to obtain realistic colorings with our model.

See the [project page](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/) for more detailed information.

## Architecture
![Network Image](https://github.com/2021rahul/colorization_tensorflow/blob/master/img/Architecture.jpg)
The Deep netwotk used can be divided into the following components:
#### - Low-level Features
It is a 6 layer convolutional neural network which obtains low-level features from the input image, which is fed to both the mid-level features features network as well as global features network.
#### - Global Image Features
Global image features are obtained by further processing the low-level features with four convolutional layers followed by three fully connected layer.
#### - Mid-level Features
The low-level features are processed with two convolutional layers to obtain the mid-level features.
#### - Fusion Layer
The global features are concatenated with the local features at each spatial location and processed with a small one layer network.
#### - Colorization Network
The fused features are processed by a set of convolutional & upsampling layers. The output layer is a convolutional layer with sigmoid transfer function that outputs the chrominance of the grayscale image.

The components are all trained in an end-to-end fashion. The chrominance is fused with the luminance to form the out

## Method
Following are the steps followed while training the network:
- Convert the colour image into grayscale and `CIEL*a*b*` colourspace.
- Pass the grayscale image as input to the model.
- Compute the MSE between target output and the output of the colorization network.
- Backpropogate the loss through all the networks(global features, mid-level features & low-level features) to update all the parameters of the model.

## Code Organization
**input_create.py**     - creates the dataset for input

**read_input.py**       - reads the dataset for input 

**build_model.py**      - builds the model

**model_train.py**      - trains the model on cpu or gpu

**model_transform.py**  - transforms a grayscale to colour image
