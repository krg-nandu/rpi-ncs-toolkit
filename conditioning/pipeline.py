
# Data capture, processing and in-the-loop system for conditioning 
# experiments with zebrafish
# Author: Lakshmi

import io, time, cv2, sys
from picamera import PiCamera
import numpy
from threading import Thread
from pytictoc import TicToc

import multiprocessing
#from multiprocessing import Queue, Pool
from Queue import Queue

from mvnc import mvncapi as mvnc

from psychopy import visual, core, event
import profile

import matplotlib.pyplot as plt

w=1920
h=1080
rate=20
fw=(w+31)//32*32
fh=(h+15)//16*16
Y=None
U=None
V=None
YUV=None
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
imdims=(416,416)


# The graph file that was created with the ncsdk compiler
graph_file_name='zbox_v3.graph'

save_q = Queue(maxsize=0)
process_q = Queue(maxsize=0)
result_q = Queue(maxsize=0)

"""
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.set_xlim([0,w])
ax.set_ylim([0,h])
ax.axis('off')
"""

def display_worker(result_q):
    mywin=visual.Window([640,480],units="pix",fullscr=False,color="white")
    white=visual.ImageStim(win=mywin,image="white.png",size=(640,480),pos=[0,0])
    plate=visual.Circle(mywin,20,lineColor="Black")
    shape=visual.Circle(mywin,5,fillColor='Red')

    white.draw()
    plate.draw()
    mywin.flip()

    while not stream_done:
	(v1,v2) = result_q.get()
	#shape.setPos([v1,v2])
	white.draw()
	plate.draw()
	shape.draw()
	mywin.flip()
 	result_q.task_done()

    mywin.close()

"""
def display_worker(result_q):
    (v1,v2) = result_q.get()
    #cv2.imshow("win",plate)
    #cv2.waitKey(1)
    ax.cla()
    ax.scatter(v1,v2)
    plt.pause(0.0001)
    result_q.task_done()
"""

def process_worker(graph, process_q, result_q):
    Xsmall = numpy.ndarray(shape=imdims,dtype=numpy.float16)
    Xsmall_int = numpy.ndarray(shape=imdims,dtype=numpy.uint8)
    Xbig = numpy.ndarray(shape=(w,h),dtype=numpy.float16)

    while True:
        frame = process_q.get()

	cv2.resize(src=frame, dsize=dim, dst=Xsmall_int, interpolation=cv2.INTER_NEAREST)
	Xsmall[:] = Xsmall_int[:]
	Xsmall /= 255.
	Xsmall -= 0.5

	ret = graph.LoadTensor(Xsmall,None)
        out, _ = graph.GetResult()

        # do some post processing
        spatial_grid = out.reshape(output_dim)
	conf = spatial_grid.squeeze()[:,:,-1]
        idx = numpy.unravel_index(numpy.argmax(conf),conf.shape)
        val = spatial_grid[idx[0],idx[1],:]
        v1 = (idx[0]*32 + val[0]*32) * (w / 416.)
        v2 = (idx[1]*32 + val[1]*32) * (h / 416.)
	print(v1,v2)
	result_q.put_nowait((v1,v2))
	process_q.task_done()


def saver_worker(save_q):
    while True:
	(X, id) = save_q.get()
 	cv2.imwrite("video/img"+str(id)+".png",X)
	save_q.task_done()

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

    plate = cv2.imread("white.png")
    cur_im = plate.copy()

    #display_thread = Thread(target=display_worker,args=(result_q,))
    #display_thread.setDaemon(True)
    #display_thread.start()
    mywin=visual.Window([640,480],units="pix",fullscr=False,color="white")
    white=visual.ImageStim(win=mywin,image="white.png",size=(640,480),pos=[0,0])
    plate=visual.Circle(mywin,20,lineColor="Black")
    shape=visual.Circle(mywin,5,fillColor='Red')

    # start the capture threads
    startcapt()

    # to avoid the initial burst of exposure
    time.sleep(2)

    # pool of threads for saving to disk
    for i in range(16):
        save_thread = Thread(target=saver_worker,args=(save_q,))
        save_thread.setDaemon(True)
        save_thread.start()

    for i in range(16):
        process_thread = Thread(target=process_worker,args=(graph,process_q,result_q))
        process_thread.setDaemon(True)
        process_thread.start()

    t = TicToc()
    t.tic()
    frame_counter = 1
    while True:
        if frame_counter%(5*rate) == 0:
	    break
	save_q.put_nowait((Y,frame_counter))
	
	if frame_counter%3 == 0:
	    process_q.put_nowait(Y)

	white.draw()
    	plate.draw()
	if result_q.qsize() > 0:
	    try:
            	(v1,v2) = result_q.get(timeout=5)
	    	shape.draw()
	    	result_q.task_done()
	    except Queue.Full:
		break
    	mywin.flip()

	# this is to ensure frame rate compatibility
	time.sleep(1./rate)
		
	# update frame index
	frame_counter += 1
    t.toc()
    stream_done=True

    #cv2.destroyAllWindows()
    #output_video.release()

    process_q.join()
    #result_q.join()
    save_q.join()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    #profile.run('main()')
    sys.exit(main())
