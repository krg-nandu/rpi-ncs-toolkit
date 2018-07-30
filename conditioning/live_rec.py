#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

# Modified to include video mode for zebrafish detection
# Author: Lakshmi

from mvnc import mvncapi as mvnc
import numpy, cv2, sys
from pytictoc import TicToc
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Queue, Pool
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import profile
#from skimage.transform import resize
from imutils.video import VideoStream

cv2.setUseOptimized(True)

dim=(416,416)
output_dim=(13,13,3)
EXAMPLES_BASE_DIR='/home/pi/rpi-ncs-toolkit/data/'
VIDEO_FULL_PATH = EXAMPLES_BASE_DIR + 'sample_video.mp4' 
IMAGE_FULL_PATH = EXAMPLES_BASE_DIR + 'sample_image_2.jpg'

__MODE__ = 'VID' # 'VID' or 'IMG'
__VERBOSE__ = False

#im = ((resized_image + 0.5) * 255.).astype(numpy.uint8)
#cv2.rectangle(im, (int(v2 - 16-1), int(v1 - 16-1)), (int(v2 + 16-1), int(v1 + 16-1)),
#              (0,0,255), 1)
#cv2.imshow("win", im)
#cv2.waitKey(1)

def worker(graph, input_q):
    while True:
        frame = input_q.get()
        graph.LoadTensor(cv2.resize(frame/255.- 0.5,dim).astype(numpy.float16),None)
        out, userobj = graph.GetResult()
        # do some post processing
        spatial_grid = out.reshape(output_dim)	
	conf = spatial_grid.squeeze()[:,:,-1]
        idx = numpy.unravel_index(numpy.argmax(conf),conf.shape)
        val = spatial_grid[idx[0],idx[1],:]
        v1 = idx[0]*32 + val[0]*32
        v2 = idx[1]*32 + val[1]*32
	print('done')
        #output_q.put((frame, (v1,v2)))

# This function is called from the entry point to do
# all the work of the program
def main():
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

    # create a camera instance with specified parameters
    # here we opt for a lower resolution since the network was trained on a much smaller dimnesion
    camera = PiCamera()
    camera.resolution = (1920, 1080)
    #camera.resolution = (1296,730)
    camera.framerate = 30
    camera.exposure_mode='off'
    camera.awb_mode='off'
    camera.awb_gains=1
    camera.shutter_speed=15000
    raw_image = PiRGBArray(camera,size=(1920, 1080))
    #raw_image = PiRGBArray(camera,size=(1296,730))

    input_q = Queue(maxsize=128)
    #output_q = Queue(maxsize=32)
    pool = Pool(8, worker, (graph, input_q))

    count = 0
    # to avoid the initial burst of exposure
    time.sleep(0.1)
    T = TicToc()
    # start capturing
    #vs = VideoStream(usePiCamera=True).start()

    T.tic()
    for frame in camera.capture_continuous(raw_image, format="rgb", use_video_port=True):
    #while True:
        #T.tic()
        #img = frame.array
        #img_copy = img.copy()
        #frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #run_inference(frame.array, graph)
        #if count%3 == 0:
	    #input_q.put_nowait(frame.array)
            #(img, results) = output_q.get_nowait()
    	#frame = vs.read()
        #cv2.imshow("win",frame.array)
	#cv2.imshow("win",frame)
	#cv2.waitKey(1)
	raw_image.seek(0)
        raw_image.truncate()
	#T.toc()
        count += 1
        if count == 30:
            break
    T.toc() 
    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()
    #vs.stop()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    #profile.run('main()',sort='time')
    sys.exit(main())
