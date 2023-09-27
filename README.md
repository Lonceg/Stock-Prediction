# Stock-Prediction

Python code which uses TensorFlow library for training neural networks used in Stock Market log return prediction.

Prepared models use Reccurent Neural Network structure, Long Short Term Memory to be exact.

Prediction of chosen amount days is made based on data from previous days in following many-to-many/many-to-one arrangements:

![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/e9512346-b670-49ed-8ce9-79d585f12278)

Preprocessing of data includes adding additional metrics such as RSI or SME, therefor at least 26 rows of data have to be provided.
Preprocessing also involves generation of time series as well as standarization.

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

Code prepares decomposition of the input data as well as a heat correlation map of the features. Also all of the metrics are plotted.

![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/875d990a-a33e-42fd-8d93-86363e6b7185)

![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/c49b43d2-0823-46f6-8bd2-1734198d3a75)

![image](https://github.com/Lonceg/Stock-Prediction/assets/92753179/7b97c237-a57a-45ae-b851-d82191293514)

Prediciton of log return proves to be extremely difficult as in stock market, past does not have to be indicator of the future.
Obviously having an algorithm which can precisely predict return with high accuracy would make one a millionaire.

Stock data used are PKN and LPP from Polish stock market.

Graph show 1 day log return prediction based on previous 7 days of pkn stock.

Test Values:

![Actual vs Predicted Values Test](https://github.com/Lonceg/Stock-Prediction/assets/92753179/a60258c3-2d01-4b46-99f6-b62485cee185)

Train Values:

![Actual vs Predicted Values Train](https://github.com/Lonceg/Stock-Prediction/assets/92753179/ac1eadbc-6c81-4e2a-bac9-b281fa7d94f3)

For simple demonstration sake, I also prepared models which prepare price predictions based solely on price. These according to my research would
not be used in proffesional forecasting due to the fact that stock prices can grow infinitely. LSTM also performs poorly when it comes to extrapolation.
Training also depended on the quality of the input data. If training set exhibited constant growth, there would be bias for that in the test set, as obviously this is what
model has been trained on.

Overall, even though below predictions look pretty (not always), apparently they are less valuable than predictions based on returns.

Price prediction LPP 7 days input and 1 day output:

Test Values:

![Actual vs Predicted Values Test](https://github.com/Lonceg/Stock-Prediction/assets/92753179/cc017bec-0e25-4732-946a-1d1576a8b116)

Train Values:

![Actual vs Predicted Values Train](https://github.com/Lonceg/Stock-Prediction/assets/92753179/d52d6c85-8c2d-4c5e-9811-ee1549b62ee0)

Price prediction LPP 20 days input and 1 day output:

Test Values:

![Actual vs Predicted Values Test](https://github.com/Lonceg/Stock-Prediction/assets/92753179/c194ea43-aeb5-4441-88c2-a4d4f3f99a5f)

Train Values:

![Actual vs Predicted Values Train](https://github.com/Lonceg/Stock-Prediction/assets/92753179/fbb7418e-23c6-4b89-ad0f-ea31eb478b22)

Price prediction LPP 30 days input and 30 day output:

Test Values:

![Actual vs Predicted Values Test](https://github.com/Lonceg/Stock-Prediction/assets/92753179/a6d8332d-e3c3-4b62-b541-7d3f603bf6f3)

Train Values:

![Actual vs Predicted Values Train](https://github.com/Lonceg/Stock-Prediction/assets/92753179/c7209daf-f357-4960-95c3-9bcdcdedfce3)

Price prediction LPP 30 days as of 23/09/2023

![Plot of the stock prediction](https://github.com/Lonceg/Stock-Prediction/assets/92753179/11bd7563-af97-4ebb-92cd-1bd3065b9b3d)

Price prediction PKN 30 days input and 30 day output:

Test Values:

![Actual vs Predicted Values Test](https://github.com/Lonceg/Stock-Prediction/assets/92753179/52fd7e22-2cfb-4a9e-a9f3-fa97bb1d4626)

Train Values:

![Actual vs Predicted Values Train](https://github.com/Lonceg/Stock-Prediction/assets/92753179/e59b9272-29d9-4547-8e6b-2bb506fdca4b)

Price prediction PKN 30 days as of 25/09/2023

![Plot of the stock prediction](https://github.com/Lonceg/Stock-Prediction/assets/92753179/6f4ce330-27c5-4b97-8823-2cfbb5d1d43e)


