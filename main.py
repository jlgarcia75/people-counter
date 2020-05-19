"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import cv2
import os
import sys
import socket
import json
import numpy as np
from datetime import datetime
import logging as log
import paho.mqtt.client as mqtt
from MediaReader import MediaReader

from argparse import ArgumentParser
from inference import Network
from sys import platform

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image, video file, directory, or integer for camera.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    parser.add_argument("-ft", "--frame_threshold", type=int, default=10,
                        help="Threshold of frames to account for drops."
                        "(10 by default)")
    
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def preprocess(image, new_shape):
    new_image = cv2.resize(image, new_shape)
    new_image = new_image.transpose((2,0,1))
    new_image = new_image.reshape(1, *new_image.shape)
    return new_image

def draw_box(image, start_point, end_point):
    box_col = (0,255,0) #GREEN
    thickness = 4
    image = cv2.rectangle(image, start_point, end_point, box_col, thickness)
    return image

def find_accuracy(results):
    # Load ground truth file. Contains a list of numbers whose indeces 
    # are frame number and the value is the number of detections in the frame
    
    truth = np.loadtxt('truth.csv', delimiter=",")
    
    # Compare the predicted detections to the ground truth detections
    # and get the percentage that are equal.
    acc = np.equal(np.array(results),truth).sum()/len(truth)*100 
    return acc

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    ######### Setup fonts for text on image ########################
    # Get and open video capture
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (10, 200) 
    fontScale = .5
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 1 px 
    thickness = 1

########################################################################   
   
    predictions = []
    total_count = 0
    count = 0
    duration = 0
    frame_count = 0
    start_frame = 0
    frames_since_last_count = 0
    inFrame = False
    infer_duration_ms  = 0
    alarm_text=""
    
    # Initialise the class
    infer_network = Network()
    
    ### TODO: Load the model through `infer_network` ###
    load_duration_ms = infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    net_input_shape = infer_network.get_input_shape()
     
    #Initialize and open media reader to read camera, video, directory of images, or a single image
    cap = MediaReader(args.input)
    
    #define upper bound for new detections
    bound_buffer = 75

    # Process frames until the video ends, or process is exited
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        num_detections = 0 ## reset num_detections in current frame to 0
        # Read the next frame
        flag, frame = cap.read()
        
        if not flag:
            break
        
        frame_count+=1
        key_pressed = cv2.waitKey(60)
      
        # Pre-process the frame
        p_frame = preprocess(frame,(net_input_shape[3], net_input_shape[2])) #cv2 frame dimensions are width, height
    
        # Perform inference on the frame
        start_time = datetime.now()
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            infer_duration_ms += (datetime.now() - start_time).total_seconds()
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            #Draw num_detections
            for box in result[0][0]:
                image_id, label, conf, x_min, y_min, x_max, y_max = box
                if conf >= args.prob_threshold:
                    num_detections+=1 ##counts num_detections in the current frame
                    width = frame.shape[1]
                    height= frame.shape[0]
                    x_min = int(x_min*width)
                    y_min = int(y_min*height)
                    x_max = int(x_max*width)
                    y_max = int(y_max*height)
                    top_left = (x_min, y_min)
                    bottom_right = (x_max, y_max)
                    frame = draw_box(frame,top_left, bottom_right)
                    if not inFrame and not y_min < bound_buffer: ## no person detected in previous frame and box not on top
                        start_frame = frame_count ##start counting frames when box first shows up
                        count+=1
                        total_count+=count 
                    inFrame = True
           
        #gather predictions for accuracy test
        predictions.append(num_detections)
         
        if inFrame:
            if (frame_count - start_frame)/10 > 17:
                    alarm_text = "TOO LONG!!! More than 17 seconds in frame."
            if num_detections == 0: ## num_detections disappeared and person is in frame so start counting frames, possible exit
                frames_since_last_count+=1 # Count frames when there is a person in the frame
                if frames_since_last_count > args.frame_threshold: 
                   inFrame = False    # No one in Frame if there have been 0 num_detections for longer than frame_threshold
                   duration = frame_count - start_frame
                   start_frame = frame_count
                   count = 0
                   frames_since_last_count = 0 ##reset frames since last count
                   alarm_text = ""
            else: ## frame contains num_detections and person is in frame so don't count frames, possible entry
                frames_since_last_count=0 
        
        frame = cv2.putText(frame, alarm_text, (70,20), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        duration_secs = duration/10
        #Add duration text to frame
        duration_text = "Duration of last person to leave frame: {} secs".format(duration_secs)
        count_text = "Total people counted so far: {}".format(total_count)
        frame = cv2.putText(frame, duration_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, count_text, (org[0],org[1]+15), font, fontScale, color, thickness, cv2.LINE_AA)
        
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###es}))
        client.publish("person", json.dumps({"count": num_detections}))
        client.publish("person", json.dumps({"total": total_count}))
        client.publish("person/duration", json.dumps({"duration": duration_secs}))
                                                          
        ### TODO: Send the frame to the FFMPEG server ###
        ### TODO: Write an output image if `single_image_mode` ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()
    outF = open("openvino_stats.txt", "a")
    now = datetime.now()
    outF.write("OpenVINO Results\n")
    outF.write ("\nCurrent date and time:\n")
    outF.write(now.strftime("%Y-%m-%d %H:%M:%S"))
    outF.write("\nPlatform: {}".format(platform))
    outF.write("\nDevice: {}".format(args.device))
    outF.write("\nModel : {}".format(args.model))
    outF.write("\nProbably Threshold: {}".format(args.prob_threshold))
    outF.write("\nAccuracy: {}%".format(find_accuracy(predictions)))
    outF.write("\nLoad Model Time: {:.2f} ms".format(load_duration_ms))
    outF.write("\nTotal Inference Time: {:.2f} ms".format(infer_duration_ms*1000))
    outF.write("\nAverage Inference Time: {:.2f} ms".format(infer_duration_ms/frame_count*1000))
    outF.write("\n\n*********************************************************************************\n\n\n")
    outF.close()
    
def main():
    """
    Load the network and parse the output.
    
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
