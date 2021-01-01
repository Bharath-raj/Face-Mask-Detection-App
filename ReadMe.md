
(PythonFD) c:\Projects\Face-Mask-Detection>pip install -r requirements.txt

Requirement already satisfied: tensorflow>=1.15.2 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from -r requirements.txt (line 1)) (2.3.0)
Requirement already satisfied: imutils==0.5.3 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from -r requirements.txt (line 3)) (0.5.3)
Collecting keras==2.3.1
  Downloading Keras-2.3.1-py2.py3-none-any.whl (377 kB)
     |████████████████████████████████| 377 kB 544 kB/s
Requirement already satisfied: keras-preprocessing>=1.0.5 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from keras==2.3.1->-r requirements.txt (line 2)) (1.1.0)
Requirement already satisfied: pyyaml in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from keras==2.3.1->-r requirements.txt (line 2)) (5.3.1)
Requirement already satisfied: h5py in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from keras==2.3.1->-r requirements.txt (line 2)) (2.10.0)
Requirement already satisfied: keras-applications>=1.0.6 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from keras==2.3.1->-r requirements.txt (line 2)) (1.0.8)
Requirement already satisfied: six>=1.9.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from keras==2.3.1->-r requirements.txt (line 2)) (1.15.0)
Collecting matplotlib==3.2.1
  Downloading matplotlib-3.2.1-cp38-cp38-win_amd64.whl (9.2 MB)
     |████████████████████████████████| 9.2 MB 2.2 MB/s
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from matplotlib==3.2.1->-r requirements.txt (line 6)) (1.3.1)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from matplotlib==3.2.1->-r requirements.txt (line 6)) (2.4.7)
Requirement already satisfied: python-dateutil>=2.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from matplotlib==3.2.1->-r requirements.txt (line 6)) (2.8.1)
Requirement already satisfied: cycler>=0.10 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from matplotlib==3.2.1->-r requirements.txt (line 6)) (0.10.0)
Collecting numpy==1.18.2
  Downloading numpy-1.18.2-cp38-cp38-win_amd64.whl (12.8 MB)
     |████████████████████████████████| 12.8 MB 13 kB/s
Collecting opencv-python==4.2.0.*
  Downloading opencv_python-4.2.0.34-cp38-cp38-win_amd64.whl (33.1 MB)
     |████████████████████████████████| 33.1 MB 1.3 MB/s
Collecting scipy==1.4.1
  Using cached scipy-1.4.1-cp38-cp38-win_amd64.whl (31.0 MB)
Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (3.1.0)
Requirement already satisfied: absl-py>=0.7.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.11.0)
Requirement already satisfied: google-pasta>=0.1.8 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.2.0)
Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (2.3.0)
Requirement already satisfied: wrapt>=1.11.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.12.1)
Requirement already satisfied: termcolor>=1.1.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.1.0)
Requirement already satisfied: astunparse==1.6.3 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.6.3)
Requirement already satisfied: grpcio>=1.8.6 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.31.0)
Requirement already satisfied: protobuf>=3.9.2 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (3.13.0)
Requirement already satisfied: wheel>=0.26 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.36.2)
Requirement already satisfied: tensorboard<3,>=2.3.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorflow>=1.15.2->-r requirements.txt (line 1)) (2.3.0)
Collecting gast==0.3.3
  Using cached gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
Collecting keras-preprocessing>=1.0.5
  Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
Requirement already satisfied: setuptools in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from protobuf>=3.9.2->tensorflow>=1.15.2->-r requirements.txt (line 1)) (51.0.0.post20201207)
Requirement already satisfied: werkzeug>=0.11.15 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.0.1)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.4.2)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.6.0)
Requirement already satisfied: requests<3,>=2.21.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (2.25.1)
Requirement already satisfied: markdown>=2.6.8 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (3.3.3)
Requirement already satisfied: google-auth<2,>=1.6.3 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.24.0)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (4.6)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.2.8)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (4.2.0)
Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (0.4.8)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (1.26.2)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (2.10)
Requirement already satisfied: chardet<5,>=3.0.2 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (3.0.4)
Requirement already satisfied: oauthlib>=3.0.0 in c:\users\bharath raj\anaconda3\envs\pythonfd\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=1.15.2->-r requirements.txt (line 1)) (3.1.0)
Installing collected packages: numpy, scipy, keras-preprocessing, gast, opencv-python, matplotlib, keras
  Attempting uninstall: numpy
    Found existing installation: numpy 1.19.2
    Uninstalling numpy-1.19.2:
      Successfully uninstalled numpy-1.19.2
  Attempting uninstall: scipy
    Found existing installation: scipy 1.5.2
    Uninstalling scipy-1.5.2:
      Successfully uninstalled scipy-1.5.2
  Attempting uninstall: keras-preprocessing
    Found existing installation: Keras-Preprocessing 1.1.0
    Uninstalling Keras-Preprocessing-1.1.0:
      Successfully uninstalled Keras-Preprocessing-1.1.0
  Attempting uninstall: gast
    Found existing installation: gast 0.4.0
    Uninstalling gast-0.4.0:
      Successfully uninstalled gast-0.4.0
  Attempting uninstall: matplotlib
    Found existing installation: matplotlib 3.3.3
    Uninstalling matplotlib-3.3.3:
      Successfully uninstalled matplotlib-3.3.3
  Attempting uninstall: keras
    Found existing installation: Keras 2.4.3
    Uninstalling Keras-2.4.3:
      Successfully uninstalled Keras-2.4.3
Successfully installed gast-0.3.3 keras-2.3.1 keras-preprocessing-1.1.2 matplotlib-3.2.1 numpy-1.18.2 opencv-python-4.2.0.34 scipy-1.4.1

(PythonFD) c:\Projects\Face-Mask-Detection>dir
 Volume in drive C is OS
 Volume Serial Number is 340F-F736

 Directory of c:\Projects\Face-Mask-Detection

01-01-2021  12:18    <DIR>          .
01-01-2021  12:18    <DIR>          ..
01-01-2021  12:18    <DIR>          dataset
01-01-2021  12:18             4,336 detect_mask_video.py
01-01-2021  12:18    <DIR>          face_detector
01-01-2021  12:18        11,483,520 mask_detector.model
01-01-2021  12:18            41,029 plot.png
01-01-2021  12:18               122 requirements.txt
01-01-2021  12:42             4,741 train_mask_detector.py
               5 File(s)     11,533,748 bytes
               4 Dir(s)  359,427,096,576 bytes free



(PythonFD) c:\Projects\Face-Mask-Detection>python train_mask_detector.py
[INFO] loading images...
C:\Users\BHARATH RAJ\Anaconda3\envs\PythonFD\lib\site-packages\PIL\Image.py:951: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
2021-01-01 12:54:03.469258: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9412608/9406464 [==============================] - 23s 2us/step
[INFO] compiling model...
[INFO] training head...
Epoch 1/20
2021-01-01 12:54:50.995754: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
2021-01-01 12:54:51.056450: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 156905472 exceeds 10% of free system memory.
 1/95 [..............................] - ETA: 0s - loss: 0.9751 - accuracy: 0.50002021-01-01 12:54:52.040991: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
2021-01-01 12:54:52.088247: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 156905472 exceeds 10% of free system memory.
 2/95 [..............................] - ETA: 30s - loss: 1.1240 - accuracy: 0.48442021-01-01 12:54:52.759136: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
95/95 [==============================] - 104s 1s/step - loss: 0.3516 - accuracy: 0.8464 - val_loss: 0.0802 - val_accuracy: 0.9831
Epoch 2/20
95/95 [==============================] - 93s 984ms/step - loss: 0.1218 - accuracy: 0.9591 - val_loss: 0.0498 - val_accuracy: 0.9896
Epoch 3/20
95/95 [==============================] - 94s 994ms/step - loss: 0.0922 - accuracy: 0.9674 - val_loss: 0.0409 - val_accuracy: 0.9896
Epoch 4/20
95/95 [==============================] - 98s 1s/step - loss: 0.0678 - accuracy: 0.9733 - val_loss: 0.0364 - val_accuracy: 0.9909
Epoch 5/20
95/95 [==============================] - 96s 1s/step - loss: 0.0625 - accuracy: 0.9782 - val_loss: 0.0326 - val_accuracy: 0.9922
Epoch 6/20
95/95 [==============================] - 95s 996ms/step - loss: 0.0554 - accuracy: 0.9812 - val_loss: 0.0302 - val_accuracy: 0.9909
Epoch 7/20
95/95 [==============================] - 96s 1s/step - loss: 0.0510 - accuracy: 0.9838 - val_loss: 0.0418 - val_accuracy: 0.9896
Epoch 8/20
95/95 [==============================] - 96s 1s/step - loss: 0.0456 - accuracy: 0.9829 - val_loss: 0.0317 - val_accuracy: 0.9909
Epoch 9/20
95/95 [==============================] - 95s 999ms/step - loss: 0.0471 - accuracy: 0.9838 - val_loss: 0.0343 - val_accuracy: 0.9909
Epoch 10/20
95/95 [==============================] - 96s 1s/step - loss: 0.0424 - accuracy: 0.9845 - val_loss: 0.0265 - val_accuracy: 0.9922
Epoch 11/20
95/95 [==============================] - 95s 998ms/step - loss: 0.0346 - accuracy: 0.9888 - val_loss: 0.0266 - val_accuracy: 0.9922
Epoch 12/20
95/95 [==============================] - 96s 1s/step - loss: 0.0295 - accuracy: 0.9895 - val_loss: 0.0265 - val_accuracy: 0.9922
Epoch 13/20
95/95 [==============================] - 96s 1s/step - loss: 0.0308 - accuracy: 0.9885 - val_loss: 0.0260 - val_accuracy: 0.9922
Epoch 14/20
95/95 [==============================] - 96s 1s/step - loss: 0.0235 - accuracy: 0.9921 - val_loss: 0.0252 - val_accuracy: 0.9922
Epoch 15/20
95/95 [==============================] - 95s 1s/step - loss: 0.0272 - accuracy: 0.9898 - val_loss: 0.0336 - val_accuracy: 0.9896
Epoch 16/20
95/95 [==============================] - 96s 1s/step - loss: 0.0261 - accuracy: 0.9921 - val_loss: 0.0310 - val_accuracy: 0.9896
Epoch 17/20
95/95 [==============================] - 103s 1s/step - loss: 0.0271 - accuracy: 0.9904 - val_loss: 0.0311 - val_accuracy: 0.9896
Epoch 18/20
95/95 [==============================] - 106s 1s/step - loss: 0.0226 - accuracy: 0.9947 - val_loss: 0.0248 - val_accuracy: 0.9922
Epoch 19/20
95/95 [==============================] - 96s 1s/step - loss: 0.0241 - accuracy: 0.9914 - val_loss: 0.0261 - val_accuracy: 0.9922
Epoch 20/20
95/95 [==============================] - 97s 1s/step - loss: 0.0224 - accuracy: 0.9927 - val_loss: 0.0223 - val_accuracy: 0.9948
[INFO] evaluating network...
              precision    recall  f1-score   support

   with_mask       0.99      0.99      0.99       383
without_mask       0.99      0.99      0.99       384

    accuracy                           0.99       767
   macro avg       0.99      0.99      0.99       767
weighted avg       0.99      0.99      0.99       767

[INFO] saving mask detector model...

(PythonFD) c:\Projects\Face-Mask-Detection>python detect_mask_video.py
2021-01-01 13:33:30.773835: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[INFO] starting video stream...
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)
(1, 1, 200, 7)