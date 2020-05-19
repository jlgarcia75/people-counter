# Project Write-Up

Due to the web gui not correctly showing the total counted and the duration of the last person in the video, I added these statistics as text within the output video. I also added a red alarm text that shows up when a person is more than 17 seconds in the video.

The executable "runapp" script contains a full command line to run the app.

## Media Input
I created a MediaReader class, in MediaReader.py, that abstracts the opening and reading of different types of input. The accepted inputs are:

1. video with .mp4 extension
2. images with extension ".jpg",".gif",".png",".tga",".bmp"
3. a directory of images
4. an integer for a webcam device.

If an invalid input is passed an exception is raised telling the user that the input is invalid or path does not exist.

## Explaining Custom Layers

The process behind converting custom layers involves...
Creating extensions to the Model Optimizer and the Inference Engine.

To create extensions for the Model Optimizer, you need to use the Customer Layer Extractor and the Customer Layer Operation. The Custom Layer Extractor identifies the custom layer operations and extracts parameters for each instance of the custom layer. The custom layer operation specifies attributes that are supported by the custom layer and computes the output shape for each instance of the custom layer from its parameters.

The Model Extension Generator is used to create extensions for the target plugin in the Inference Engine that requires support for the custom layers. The Model Extension Generator Tool creates source file templates with functions that the user must implement. The resulting extension is a library file that is loaded by the device plugin when the Inference Engine runs.

Some of the potential reasons for handling custom layers are:
Custom layers are layers that are not supported by OpenVINO for a given framework--i.e. Tensorflow, Caffe, MXNet, Kaldy, ONNX-- or the target device. For example, you may have a model whose layers are all supported by the CPU plugin but some of the layers are not supported on VPU. If you want to run this model on a VPU, you must implement custom layers for the unsupported layers. The user must implement custom layers to tell the Model Optimizer and the Inference Engine how to handle these unsupported layers.


# Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

I did all performance tests on my local machine:
iMac, Late 2015
Processor: 3.2 GHz Quad-Core Intel(R) Core(TM) i5
Memory: 16 GB
Graphics: AMD Radeon R9
macOS Catalina version 10.15.4

##Tensorflow
Since I am not experienced with Tensorflow, I used this public tutorial, Adapted from: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb, as a base and adapted it for my needs. The code I used is in file tf_test.py.

I decided to use the model ssd_mobilenet_v2_coco_2018_03_29 with a probability threshold of .3, which means that any detections with a confidence of less than .3 where thrown out.

In order to compare the accuracy of the models I had to establish a Ground Truth. This was my method for establishing Ground Truth:
1. Wrote a python script to extracted all the frames from the original video into individual jpg files in a directory.
2. The script named each file with the number of the frame, i.e. frame_##.jpg.
3. With enlarged icons, I viewed all the frames and copied the frames without people in them into a separate directory called "noperson".
4. Wrote another script that looks at the names of all the files in the "noperson" directory, finds the number in the file names, and populates a list whose indeces are the frame numbers and values are the number of people in that frame. In our case, that number is always 1.
5. The script then writes a file with the list of 0s and 1s, with 0s in locations without people in the frames, and 1 where there is a person in the frame.
6. The inferencing script for each model then reads this file and imports the list as the "truth" that gets compared to the model predictions.
7. In the main script, as each frame gets processed, the number of detections gets appended to a list that gets compared to the "truth" list.
8. The number of matches get summed and divided by the number of frames to arrive at the accuracy percentage.

To measure the inference time, I used the datetime.now() method to capture the amount of time to run the only the actual inference commands. I aggregated these times and took the average at the end.
______________________________________________________________________
|                     |   Tensorflow 1.15.0 | OpenVINO          | Diff   
| Accuracy %          |       94.33         |  93.62            |  .71   
| Model Size          |      PB  69.7 MB    | IR + XML 67.41 MB |  2.29 MB
| Avg. Inference Time |          145 ms     |  26 ms            |  119 ms

The original model in Tensorflow is only slightly more accurate. The original model size is slightly bigger than before it was converted into the Intermediate Representation with the Model Optimizer. The inference time with OpenVINO was much faster than with Tensorflow.

Each time the app is run, statistics on accuarcy, load model time, total inference time, and average inference time are appended to a file called openvino_stats.txt. You can find statistics from running the Tensorflow model in tf_stats.txt.

NOTE ON DURATION: There is something strange happening with the duration counter when I run the app in the workspace. It works perfectly on my local machine but there is a slight inaccuracy when the second person is on the scene.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1. ATM Kiosk - count and track how many people come into an ATM kiosk. If more than one person is detected, the system may issue a warning or shutdown the ATM. This is useful as a security precaution. The data from multiple kiosks can be aggregated to track usage, demand, etc.
2. Public Transportation Passenger Counter - The solution can count the number of people entering a public transportation vehicle. When people counter data is aggregated from multiple vehicles, it can be used to predict demand for transportation across the different vehicles, areas, etc. The data can also be used immediately to limit the number of people that enter a vehicle.
3. Retail - the usages in retail are limitless. A store can count people that enter certain areas to determine how well signage and marketing are working in those areas. The data can be used to compare traffic to actual sales. People can be counted at entry to the store, entry into departments, and entry into smaller areas, such as a particular aisle.

Each of these cases is useful because the almost realtime availability of traffic data allows enterprises to make decisions about changes needed to accomodate or improve traffic patterns in targeted areas. Otherwise, non-digital ways of counting people require large delays in gathering data and decision-making.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
The environment in which a model is deployed affects the performance and effectiveness of the model on a targeted edge deivce. You must use a model that has been trained with data captured in environments similar to the target environment.

The available lighting can make make images and objects in images either too dark or too bright, which can affect the ability of the model to detect the objects. For example, if the solution will deployed in sunlight then it will require a different model or training data than a solution that is deployed indoors under flourescent light.

The model accuracy must be taken into account because it affects how the computing device performs and the usefulness of the data. A high accuracy with high-precision, i.e. FP32 vs UINT8, will require more compute power or more compute time. The edge device must be able to handle the computational load imposed by the model accuracy requirements. If you have to use a model with low accuracy because your edge device does not have enough resources then this may cause the data to be unusable.

The focal length of the camera determines the placement of the camera in relation to the scene being captured. The camera must be placed so that the objects being detected are of adequate size to be detected by the model. Again, the model used must have been trained with data captured under similar circumstances as the target environment, which includes camera placement.

The input image size affects memory requirements. Edge devices are resource constrained and are usually limited by the amount of local memory. Since all processing must be done locally on the device, the amount of memory in the device must be able to hold the input images, output images, code, and all other data required to perform inference.
