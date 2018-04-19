# Gesture Action Recognition
Fine-tune the pretrained CNN models (AlexNet, VGG, ResNet) followed by LSTM. This network is applied on gesture controlled drone. 

### Training:

- Download the Helicopter Marshalling Dataset: https://drive.google.com/file/d/1xwwt461qCQ5WQiHve97QghfwfdO6aW-U/view?usp=sharing
- Put the dataset under the '/data' folder
- Run the training code and specify the path to the data folder
'''
python basic_lstm.py ../data
'''


### Testing:

- Run the online testing code using webcam with specified model:
```
cd testing
python lstm_test.py ../weights/model_best_865.pth.tar 
```

### Dependencies:
- pyTorch-0.3.xx
- Opencv-3.3.1
- PIL-5.0.0
- Numpy-1.13.1

