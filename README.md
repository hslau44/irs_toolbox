# Self-Supervised WiFi-Based Activity Recognition

*Andy Lau*

# Introduction
This repository is my recent work for Self-Supervised Wi-Fi-based Activity Recognition, it is designed based on the template provided by Francesco Saverio Zuppichini's work *a PyTorch Deep Learning Template for the computer vision project*.

Wi-Fi-Based Activity Recognition recognise and detect the changing patterns on the Wi-Fi signal induced by human activity and the environment to perform its task. With the device-less nature, Wi-Fi-based Activity Recognition prevents the problems during healthcare monitoring, i.e. the user/patient may forget/resist wearing; and because it does not depend on the optical sensor, it is suitable for various lighting condition and more privacy preserved. Under the Wi-Fi-based Activity Recognition, the most popular method is the use of Channel State Information (CSI) because of its fine grain resolution and accessablity, i.e., it can be extracted by the commercially available network card (e.g. Intel® Wi-Fi Link 5300).

Wi-Fi-based Activity Recognition mainly consists of three stages: Data collection, Signal Processing and Signal Classification. Raw Wi-Fi signal is inherently noisy, and the early research employed a significant effort in signal processing so that the statistical model can recognise the cleaned signal. The research is then moving toward using machine learning algorithm and has gained significant progress for deploying system under more complex indoor environments. With the usage of Deep Learning, the field has become moving toward end-to-end system design.

In the current stage, we treat the signal classification as a computer vision task, i.e., treating signal as image, and use Convolutional Neural Network. With only 3 convolutional layers and less than 400,000 parameters, the network can work reasonably well with the raw data. But Even with the learning power of Deep Learning, there are several challenges, such as the high demand for labelled samples to train the network, and we still need more work to improve the model performance under complex environments in order to make it viable for real-world scenario. The goal of this project is to tackle these two problems with a popular technique in computer vision called self-supervised contrastive learning.

## Instruction

Due to the increase size of data, instead of loading all data into the memory, it loads data with following process:

1. extract all filepaths into a dataframe using `data.utils.filepath_dataframe`
2. split the dataframe into multiple dataframes with `data.selection.Selection`
3. create corresponding dataloaders with `data.torchData.DataLoading`

For instruction, please run the notebook `demo.ipynb`

Three notebook are available in laboratory:
- `Standard.ipynb`: Normal Supervised Learning
- `CrossValidation.ipynb`: Cross Validation
- `Contrastive_Learning.ipynb`: Contrastive Learning NT-Xent (Chen et al. 2020) with fine tuning


## Structure
```
.
├── callbacks // here you can create your custom callbacks
├── checkpoint // were we store the trained models
├── data // quick data import and transformation pipeline
│ ├── selection // data selection for train-validation-test split
│ ├── transformation // custom transformation compatible with torchvision.transform
│ ├── torchData // custom torch Dataset and DataLoader
│ ├── custom_data.py // data for specific format
│ ├── load_npy_format.py // tranform and load csv files into npy files
│ └── utils.py
├── laboratory // store the trained models and record
├── losses // custom losses
├── metrics // custom metrics
├── main.py
├── models // quick default model setup
│ ├── baseline.py // baseline models
│ ├── cnn.py // torchvision models
│ ├── self_supervised.py // torch.nn.module for contrastive learning
│ ├── temproal.py // CNN-LSTM
│ └── utils.py // utility torch.nn.module
├── playground.ipynb // fast experiment with things
├── contrastive_learning.ipynb // the notebook version of main.py
├── README.md
├── test // to be implemented
└── utils.py // utilities functions
```
