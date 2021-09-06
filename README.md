# Self-Supervised WiFi-Based Activity Recognition

*Andy Lau*

# Introduction
This repository is my recent work for Self-Supervised Wi-Fi-based Activity Recognition, it is designed based on the template provided by Francesco Saverio Zuppichini's work *a PyTorch Deep Learning Template for the computer vision project*.

Wi-Fi-Based Activity Recognition recognise and detect the changing patterns on the Wi-Fi signal induced by human activity and the environment to perform its task. With the device-less nature, Wi-Fi-based Activity Recognition prevents the problems during healthcare monitoring, i.e. the user/patient may forget/resist wearing; and because it does not depend on the optical sensor, it is suitable for various lighting condition and more privacy preserved. Under the Wi-Fi-based Activity Recognition, the most popular method is the use of Channel State Information (CSI) because of its fine grain resolution and accessablity, i.e., it can be extracted by the commercially available network card (e.g. Intel® Wi-Fi Link 5300).

Wi-Fi-based Activity Recognition mainly consists of three stages: Data collection, Signal Processing and Signal Classification. Raw Wi-Fi signal is inherently noisy, and the early research employed a significant effort in signal processing so that the statistical model can recognise the cleaned signal. The research is then moving toward using machine learning algorithm and has gained significant progress for deploying system under more complex indoor environments. With the usage of Deep Learning, the field has become moving toward end-to-end system design.

In the current stage, we treat the signal classification as a computer vision task, i.e., treating signal as image, and use Convolutional Neural Network. With only 3 convolutional layers and less than 400,000 parameters, the network can work reasonably well with the raw data. But Even with the learning power of Deep Learning, there are several challenges, such as the high demand for labelled samples to train the network, and we still need more work to improve the model performance under complex environments in order to make it viable for real-world scenario. The goal of this project is to tackle these two problems with a popular technique in computer vision called self-supervised contrastive learning.

## Instruction
The repository is intended to works as a package, which includes the data processing, ETL pipeline, CNN models and loss function. We reduce certain functions from the template for simplicity.

For demonstration, please run the following commands in the project's root directory, or run the jupyter notebook `contrastive learning.ipynb` `./main.py {trainmode}{network}{pairing}{t}`
- `trainmode`: for standard non-pretrained training, type `normal`; for contrastive pretraining, type `simclr`.
- `network`: four network architectures are available, shallow three layer CNN model `shallow`, AlexNet (Krizhevsky et al. 2017) `alexnet`, ResNet18 (He et al. 2015) `resnet` and VGG16 (Simonyan 2014) `vgg16`.
- `pairing`: self supervised contrastive learning uses a pair of representation of the sample which has same contextual meaning, in the experiment, we use either CSI collected from different view `nuc2` or Passivie Wi-Fi Radar `pwr` along with the main CSI.
- `t`: we use Normalized temperature cross Entropy (NT-Xent) to calculate the contrastive loss, temperature is to control the smoothness, the available value are 0.1 `L`, 0.5 `M`, and 1.0 `H`.

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
