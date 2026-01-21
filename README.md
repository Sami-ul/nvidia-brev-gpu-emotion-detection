# Training an Emotion Detection Convolutional Neural Network using NVIDIA's Brev Platform

## Neural Network

The data used was from Kaggle, `data.py` details how to download it. The dataset contains a test and train set of various images of human faces and is labeled by emotions. PyTorch's ImageFolder library makes it very easy to convert this into a usable dataset.

The training script uses PyTorch and sets up transforms, convolutional layers and a classifier layer. We use a batch size of 128 since an NVIDIA T4 GPU can handle it and can speed up our training significantly. We can also have number of workers for data loading be 4, since we are using Brev. The transforms are applied to the dataset to "generate" more data by applying natural transformations. For example, mirroring a face about the y axis does not modify the emotion shown on the face. The convolutional layers simplify the image down to its most critical features, while the classification layer converts the 2d image into a 1d tensor of numbers and outputs one of 7 emotion classes.


## NVIDIA Brev Platform

1. Open the Brev console and ensure you have credits, navigate to the T4 GPU option. [Create a new T4 environment here.](https://brev.nvidia.com/environment/new?gpu=T4)

2. Select one of the GPUs and start the instance in VM mode, this project will not need Jupyter.

3. You will need wsl, if you already have it installed you can just type `wsl` into your terminal. Once the Brev instance is running.