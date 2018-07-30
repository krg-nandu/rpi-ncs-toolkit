
# Data capture, processing and in-the-loop system for conditioning 
# experiments with zebrafish
# Author: Lakshmi

import io, time, cv2, sys
from picamera import PiCamera
import numpy
from threading import Thread
from pytictoc import TicToc

import multiprocessing
from multiprocessing import Queue, Pool

from mvnc import mvncapi as mvnc

w=640
h=480
rate=30
fw=(w+31)//32*32
fh=(h+15)//16*16
Y=None
camera=PiCamera()
camera.resolution = (w, h)
camera.framerate = rate
camera.exposure_mode='off'
camera.awb_mode='off'
camera.awb_gains=1
camera.shutter_speed=15000
stream_done=False
cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
dim=(416,416)
output_dim=(13,13,3)

# The graph file that was created with the ncsdk compiler
graph_file_name='zbox.graph'

# Open a video for writing
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#output_video = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))

def diff(frame1,frame2):
    return int(cv2.sumElems(frame1)[0]-cv2.sumElems(frame2)[0])

def display_worker(display_q):
    while True:
        (frame, (v1,v2)) = display_q.get()
        print(v1,v2)
	#cv2.imshow("win",frame.astype(numpy.uint8))
        #cv2.waitKey(1)

display_q = Queue(maxsize=128)
pool2 = Pool(4, display_worker, (display_q,))

def inference_worker(graph, input_q):
    #t = TicToc()
    while True:
        frame = input_q.get()
	#t.tic()
	X = cv2.resize(frame/255.- 0.5,dim,cv2.INTER_NEAREST)
	X = X.astype(numpy.float16,copy=False)
	#ret = mvnc.mvncStatus.BUSY
	#while ret == mvnc.mvncStatus.BUSY:
        ret = graph.LoadTensor(X,None)
        out, _ = graph.GetResult()
        # do some post processing
        spatial_grid = out.reshape(output_dim)
	conf = spatial_grid.squeeze()[:,:,-1]
        idx = numpy.unravel_index(numpy.argmax(conf),conf.shape)
        val = spatial_grid[idx[0],idx[1],:]
        v1 = idx[0]*32 + val[0]*32
        v2 = idx[1]*32 + val[1]*32
	display_q.put_nowait(((X+0.5)*255.,(v1,v2)))
	
	#t.toc()
	#print('done')
        #output_q.put((frame, (v1,v2)))

def saver_worker(save_q):
    global output_video
    while True:
	(X, id) = save_q.get()
	#output_video.write(X)
 	cv2.imwrite("video/img"+str(id)+".png",X)

def outputs():
    global Y
    stream = io.BytesIO()
    while not stream_done:

        yield stream

        stream.seek(0)
        Y=numpy.fromstring(stream.getvalue(),dtype=numpy.uint8,count=fw*fh).\
           reshape((fh,fw))

        stream.seek(0)
        stream.truncate()

def capt():
    camera.capture_sequence(outputs(), 'yuv', use_video_port=True)
    pass

def startcapt():
    t = Thread(target=capt, args=())
    t.daemon = True
    t.start()

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

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)
    graph.SetGraphOption(mvnc.GraphOption.ITERATIONS,1)
    #graph.SetGraphOption(mvnc.GraphOption.DONT_BLOCK,1)

    process_q = Queue(maxsize=32)
    pool = Pool(16, inference_worker, (graph, process_q))

    save_q = Queue(maxsize=128)
    pool1 = Pool(64, saver_worker, (save_q,))

    # start the capture threads
    startcapt()
    
    # to avoid the initial burst of exposure
    time.sleep(0.5)

    frame_counter = 1
    t=TicToc()
    t.tic()
    
    while True:
        if frame_counter%(10*rate) == 0: break
        #cv2.imshow('detection',Y)
        #cv2.waitKey(1); # != ord("q")
        save_q.put_nowait((Y,frame_counter))
	frame_counter += 1
	if frame_counter%5 == 0:
	    #print('*')
	    process_q.put_nowait(Y)
            #(img, results) = output_q.get_nowait()

	# this is to ensure frame rate compatibility
        time.sleep(1./rate)

    t.toc()
    sys.stdout.flush()
    time.sleep(1)
    stream_done=True
    cv2.destroyAllWindows()

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()
    #output_video.release()

    # wait for queue to be empty

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
