# pytorch-Realtime-facial-expression-recognition.


## Work is done for fun, so don't ask to fork it and use it for your purposes.



## Trained Model

Trained by FER2013 dataset.

* Private Data : 66%
* Public Data : 64%

Here is the result of sample image.  
Emotion | Probability | Guided Backprop | Grad-Cam | Guided Grad-Cam

<img src="./test/guided_gradcam.jpg">

## Retrain

1. see [here](./dataset/README.md) to prepare dataset.

2. execute train.py
```
cd src
python train.py
python check.py  #check.py supports cpu only
```

## Reference

* [Grad-CAM](https://github.com/kazuto1011/grad-cam-pytorch)
* [Data Augmentation / Optimizer](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)