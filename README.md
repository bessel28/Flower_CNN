# Computer vision classification and segmentation

This project trains and evaluates 2 Convolution Neural Networks, 1 for classification of flower species and 1 for segmentation of flower petal pixels from the background.

## How to run

The project exists in two forms:

1) Matlab (*.m) found at [Matlab src](src/matlab/)
1) Live (*.mlx) found at [Matlab src](src/live/)

Open the file in matlab and press `Run`

### Loading pre-trained models

Each source file has a commented line similar to: 
```matlab
% load('classnet.mat', 'classifier')
```
This can be used in the live files to skip the training stage and instead load a pre-trained model for evaluation and usage.

## Requirements

Matlab Computer Visual Toolbox
Matlab Deep Learning Toolbox
Matlab Image Processing Toolbox
Matlab Parallel Computing Toolbox
Matlab Statistics and Machine Learning Toolbox

## Report

Full scientific report can be found at [Report.pdf](Report.pdf)
