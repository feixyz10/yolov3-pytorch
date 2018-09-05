# yolov3-pytorch

### Usage (inference)

Download pre-trained weights form [here (236MB)](https://pjreddie.com/media/files/yolov3.weights) for **yolov3** or [here (34MB)](https://pjreddie.com/media/files/yolov3-tiny.weights) **yolov3-tiny** and save them to the *weights* subdirectory.  Run following command to convert them to pytorch format:

``` 
python helpers.py --model yolov3 [or yolov3-tiny]
```

Run the detector:

```
python detect.py --model yolov3 --weight weights/yolov3.pth --image data/person.jpg --thres 0.5
```

You will get prediction like this:

![](https://github.com/feixyz10/yolov3-pytorch/blob/master/data/person_prediction.jpg)

### Train

Under construction ...
