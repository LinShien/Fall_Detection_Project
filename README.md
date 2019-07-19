# Fall Detection with Co-occurrence-Feature-Learning-From-Skeleton-2D-Data

## Detail
This is a project implemented with Keras based on the code from [here](https://arxiv.org/abs/1804.06055).

Train data and Testing data is in [data_seq dirctory](https://github.com/LinShien/Fall_Detection_Project/tree/master/data_seq).

The whole model is defined in [model.py](https://github.com/LinShien/Fall_Detection_Project/blob/master/model.py).

Addtional data-processing code is in [utils.py](https://github.com/LinShien/Fall_Detection_Project).

To test the model, use [fall_detection.py](https://github.com/LinShien/Fall_Detection_Project/blob/master/fall_detection.py).

且部分資料由熱成像圖轉換成座標圖來做訓練，如下 :
class 9
![fall_img](./img/fall1.gif)

![fall_img2](./img/fall2.gif)

class 10
![sit_img](./img/sit1.gif)
![sit_img2](./img/sit2.gif)

## Result
Train with 416 skeleton sequences (20 frames)
Test with 83 skeleton sequences (20 frames)

Training acc : 100 %
Testing acc  : 94 %