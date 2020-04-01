# pytorch-Realtime-facial-expression-recognition.


## Work is done for fun, so don't ask to fork it and use it for your purposes.

# Real-Time Sample Results
<!-- blank line -->
<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="runs/emotion.PNG">
    <source src="./runs/human_emotion_recog_realtime.mp4" type="video/mp4">
  </video>
</figure>
<!-- blank line -->

# Dataset
Datset can be downloaded from kaggle (link below) and copy the extracted `fer2013.csv` file in folder fer2013.
[Dataset Kaggle Link]( https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 

# 1. Usage
*Install Library*
`Pytorch >= 1.0`<b> 
`OpenCV`<b>
`scikit-learn` <b>
`PIL` <b>
`matplotlib`<b>
`numpy`<b>
  `os`<b>


# 2. Use Command Line

*Cross entropy loss is used to train the resnet artitecture.*
Command Line execution for the realtime testing can be done bby executing-
```
python emotion_detector.py --cascade ./visualize/haarcascade_frontalface_default.xml \
--model ./public_model_414_55.pt
```

If you use SSIM method, you have to pip install SSIM-PIL.