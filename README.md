# Training an Emotion Detection Convolutional Neural Network using NVIDIA's Brev Platform

## Neural Network

The data used was from Kaggle, `data.py` details how to download it. The dataset contains a test and train set of various images of human faces and is labeled by emotions. PyTorch's ImageFolder library makes it very easy to convert this into a usable dataset.

The training script uses PyTorch and sets up transforms, convolutional layers and a classifier layer. We use a batch size of 64 since an NVIDIA T4 GPU can handle it and can speed up our training significantly. We can also have number of workers for data loading be 2, since we are using Brev. The transforms are applied to the dataset to "generate" more data by applying natural transformations. For example, mirroring a face about the y axis does not modify the emotion shown on the face. The convolutional layers simplify the image down to its most critical features, while the classification layer converts the 2d image into a 1d tensor of numbers and outputs one of 7 emotion classes.

Dataset statistics:

```
Classes: ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

Training samples: 28709

Test samples: 7178
```


## NVIDIA Brev Platform

1. Open the Brev console and ensure you have credits, navigate to the T4 GPU option. [Create a new T4 environment here.](https://brev.nvidia.com/environment/new?gpu=T4)

2. Select one of the GPUs and start the instance in VM mode, this project will not need Jupyter. You will name it here and use this name later.

3. You will need wsl, if you already have it installed you can just type `wsl` into your terminal. Once the Brev instance is running, you can open `wsl` and install the brev cli `sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"`

4. You can then run `brev login` and log in. Then you can do `brev shell <gpu-name>`.

- When we do `brev ls` we can see that we are indeed running on gpu:
```shell
emotion_detection_neural_network$ brev 
ls
You have 1 instances in Org NCA-52c9-24166
 NAME                 STATUS   BUILD      SHELL  ID         MACHINE
 emotion-detector-t4  RUNNING  COMPLETED  READY  ur5syn8q7  n1-standard-1:nvidia-tesla-t4:1 (gpu)
 ```

5. Then, you must set up a SSH key on the server with: `ssh-keygen -t ed25519 -C "your-email@example.com"`. Copy the `cat ~/.ssh/id_ed25519.pub` and paste it in GitHub > Settings > SSH and GPG keys > new SSH key.

6. You can then clone the repository the code is on using ssh `git clone git@github.com:Sami-ul/nvidia-brev-gpu-emotion-detection.git` and navigate to the directory with `cd nvidia-brev-gpu-emotion-detection`.

7. Install dependencies with `pip install -r requirements.txt`

8. Download the dataset using `python3 data.py`

9. Train the neural network with `python3 train.py`