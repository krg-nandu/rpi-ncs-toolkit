#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

# Modified to include video mode for zebrafish detection
# Author: Lakshmi

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
from pytictoc import TicToc
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Queue, Pool
import profile

dim=(416,416)
output_dim=(13,13,3)
EXAMPLES_BASE_DIR='/home/pi/rpi-ncs-toolkit/data/'
VIDEO_FULL_PATH = EXAMPLES_BASE_DIR + 'sample_video.mp4' 
IMAGE_FULL_PATH = EXAMPLES_BASE_DIR + 'sample_image_2.jpg'

__MODE__ = 'IMG' # 'VID' or 'IMG'
__VERBOSE__ = False

# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************
LABELS = ('background','zebrafish')

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, graph):

    # get a resized version of the image that is the dimensions
    # our network expects
    resized_image = preprocess_image(image_to_classify)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = graph.GetResult()
    
    spatial_grid = output.reshape(output_dim)
    if __VERBOSE__:
    	print('total num boxes: ' + str(num_valid_boxes))
    conf = spatial_grid.squeeze()[:,:,-1]
    idx = numpy.unravel_index(numpy.argmax(conf),conf.shape)
    val = spatial_grid[idx[0],idx[1],:]
    v1 = idx[0]*32 + val[0]*32
    v2 = idx[1]*32 + val[1]*32
    
    #im = ((resized_image + 0.5) * 255.) #.astype(numpy.uint8)
    #cv2.rectangle(im, (int(v2 - 16-1), int(v1 - 16-1)), (int(v2 + 16-1), int(v1 + 16-1)),
    #              (0,0,255), 1)
    #cv2.imshow("win", im)
    #cv2.waitKey(0)

# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 0

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 416
    NETWORK_HEIGHT = 416
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img/255. - 0.5
    #img = img * 0.007843
    #img = img.transpose((2,0,1))
    return img

def worker(graph, input_q, output_q):
    while True:
        frame = input_q.get()
        graph.LoadTensor(frame.astype(numpy.float16),None)
        out, userobj = graph.GetResult()
        # do some post processing
        output_q.put((frame, out))	

# This function is called from the entry point to do
# all the work of the program
def main():
    # name of the opencv window
    cv_window_name = "SSD MobileNet - hit any key to exit"

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = 'zbox.graph'

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)
    graph.SetGraphOption(mvnc.GraphOption.ITERATIONS,1)

    input_q = Queue(maxsize=32)
    output_q = Queue(maxsize=32)
    pool = Pool(8, worker, (graph, input_q, output_q))

    t = TicToc()
    if __MODE__ == 'VID':
    	vid = cv2.VideoCapture(VIDEO_FULL_PATH)
	while ( vid.isOpened() ):
		#t.tic()
		ret, frame = vid.read()
		
		#t.toc()
	        # read the image to run an inference on from the disk
	        #infer_image = cv2.resize(frame,(512,512))
		#infer_image = cv2.resize(frame,(416,416))/255.
		#input_q.put(infer_image)
		#(frame, out) = output_q.get()

	        run_inference(frame, graph)
	        #graph.LoadTensor(infer_image.astype(numpy.float16), None)
    		#t = TicToc()
    		#t.tic()
    		#output, userobj = graph.GetResult()
    		
		#t.toc()
	        # display the results and wait for user to hit a key
	        #cv2.imshow(cv_window_name, frame)
	        #cv2.waitKey(1)
    elif __MODE__ == 'IMG':
	infer_image = cv2.imread(IMAGE_FULL_PATH)
	run_inference(infer_image, graph)
	# display the results and wait for user to hit a key
        
    else:
	print ("Please specify a valid option for processing ('VID' or 'IMG')")

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    profile.run('main()')
    #sys.exit(main())
