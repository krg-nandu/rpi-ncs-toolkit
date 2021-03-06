
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

from psychopy import visual, core, event

w=1920
h=1080
rate=30
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
imdims=(416,416,3)

# The graph file that was created with the ncsdk compiler
graph_file_name='zbox_v3.graph'

def display_worker(display_q):
    mywin=visual.Window([640,480],units="pix",fullscr=False,color="white")
    white=visual.ImageStim(win=mywin,image="white.png",size=(640,480),pos=[0,0])
    plate=visual.Circle(mywin,20,lineColor="Black")
    shape=visual.Circle(mywin,5,fillColor='Red')
    while (not stream_done):
	(v1,v2) = display_q.get()
	#print(v1,v2)
	shape.setPos([v1,v2])
	white.draw()
	plate.draw()
	shape.draw()
	mywin.flip()
    
    mywin.close()
    core.quit()

def inference_worker(graph, input_q, display_q):
    #global stream_done
    Xsmall = numpy.ndarray(shape=imdims,dtype=numpy.float16)
    Xsmall_int = numpy.ndarray(shape=imdims,dtype=numpy.uint8)
    Xbig = numpy.ndarray(shape=(w,h,3),dtype=numpy.float16)

    while (not stream_done): #or (not input_q.empty()):
        frame = input_q.get()

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
	#print(v1,v2)
	display_q.put_nowait((v1,v2))

def saver_worker(save_q):
    #global stream_done
    while (not stream_done) or (not save_q.empty()):
	(X, id) = save_q.get()
	#output_video.write(X)
 	cv2.imwrite("video/img"+str(id)+".png",X)

def outputs():
    global Y
    global U
    global V
    global YUV
    stream = io.BytesIO()
    #t = TicToc()
    while not stream_done:

        yield stream

        stream.seek(0)

        Y=numpy.fromstring(stream.getvalue(),dtype=numpy.uint8,count=fw*fh).\
           reshape((fh,fw))
	U=numpy.fromstring(stream.getvalue(),dtype=numpy.uint8,count=(fw//2)*(fh//2)).\
	   reshape((fh//2,fw//2)).\
 	   repeat(2,axis=0).repeat(2,axis=1)
	V=numpy.fromstring(stream.getvalue(),dtype=numpy.uint8,count=(fw//2)*(fh//2)).\
           reshape((fh//2,fw//2)).\
           repeat(2,axis=0).repeat(2,axis=1)
	YUV=numpy.dstack((Y,U,V))[:h,:w,:]
	YUV[:,:,0] = YUV[:,:,0] - 16
	YUV[:,:,1:] = YUV[:,:,1:] - 128
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
    global stream_done
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

    process_q = Queue()
    display_q = Queue()
    #infer_thread = multiprocessing.Process(target=inference_worker,args=(graph,process_q))
    infer_thread = Thread(target=inference_worker,args=(graph,process_q,display_q))

    save_q = Queue()
    #save_thread = multiprocessing.Process(target=saver_worker,args=(save_q,))
    save_thread = Thread(target=saver_worker,args=(save_q,))

    #stim_thread = multiprocessing.Process(target=display_worker,args=(display_q,))
    #stim_thread = Thread(target=display_worker,args=(display_q,))

    # start the inference workers
    infer_thread.start()
    # start the saver workers
    save_thread.start()
    # start the stimulus display part
    #stim_thread.start()
    # start the capture threads
    startcapt()

    mywin=visual.Window([640,480],units="pix",fullscr=True,color="white")
    white=visual.ImageStim(win=mywin,image="white.png",size=(640,480),pos=[0,0])
    shape=visual.Circle(mywin,5,fillColor='Red')
    plate=visual.Circle(mywin,110,lineColor="Black")


    white.draw()
    plate.draw()
    mywin.flip()

    # to avoid the initial burst of exposure
    time.sleep(10)

    frame_counter = 1
    t=TicToc()
    t.tic()

    while True:
	print(frame_counter)
        if frame_counter%(5*rate) == 0:
	    break

        save_q.put_nowait((YUV,frame_counter))
	frame_counter += 1
	if frame_counter%5 == 0:
	    process_q.put_nowait(YUV)
	
	if not display_q.empty():
	    (v1,v2) = display_q.get()
	    shape.pos = (v2,v1)
	    white.draw()
	    plate.draw()
	    shape.draw()
	    mywin.flip()

	# this is to ensure frame rate compatibility
        time.sleep(1./rate)

    t.toc()

    stream_done=True
    cv2.destroyAllWindows()

    # make sure no more entries are added to the queue
    process_q.close()
    save_q.close()
    
    # wait for threads to complete
    infer_thread.join()

    display_q.close()
    #stim_thread.join()

    save_thread.join()

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()
    #output_video.release()

    #mywin.close()
    #core.quit()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
