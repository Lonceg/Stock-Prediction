# Stock-Prediction

Python code which uses TensorFlow library for training neural networks used in Stock Market close price predictions.

Prepared models Reccurent Neural Network structure, in particular LSTM in 3 variants:
-LSTM
-Bidirectional LSTM
-Encoder Decoder LSTM

LSTM node:
![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/ab450dc4-2ad1-4a51-8a4e-2664f104e697)

Prediction of 30 days is made based on data regarding 50 previous days in following many-to-many arrangement:
![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/9c82b80d-0c93-46df-9531-71e240237006)

Preprocessing of data includes adding additional metrics such as RSI or SME, therefor at least 76 rows of data have to be provided 
for model prediction due to the need of a window to calculate these metrics

This code uses NVIDIA libraries for accelerated training with the use of GPU. GPU used: RTX 2060.

Comparision of GPU vs CPU:
![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/b605b20f-19c5-4335-957b-06f0cb386877)

Python version: 3.9.13
Tensorflow version: 2.10
CUDA Toolkit version: 11.2
cuDNN version: 8.11
Visual C++: 2017

![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/013805f6-8ab4-4c08-a590-eef902e52d71)

Downloading NVIDIA files requires an account (for one of these)
cuDNN available under this link: https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64
CUDA available under this link: https://developer.nvidia.com/rdp/cudnn-archive

Running cuDNNN requires correct C++ compiler, all version available here: https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html
