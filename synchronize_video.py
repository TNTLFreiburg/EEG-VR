import numpy as np
import matplotlib.pyplot as plt
import cv2
import xdf

def synchronize_video(xdf_file, vid):
    """ Predicts LSL time for every frame in a corresponding video. For use in a Notebook, preceed the command with "%matplotlib qt".
    
    Args:
        xdf: Path to the .xdf file of the experiment.
        vid: Path to the video file of the experiment
        
    Returns:
        frame_time: Array of LSL times. Index corresponds to frame in the video.
    
    """
    
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    
    #
    f,a = plt.subplots()
    a.imshow(frame, cmap = 'gray')
    pos = []
    
    print('Click on the indicator and close the image.')
    
    def onclick(event):
        pos.append([event.xdata,event.ydata])
        
    f.canvas.mpl_connect('button_press_event', onclick)
    f.suptitle('Click on the indicator')
    
    plt.show(block = True)
    pos = np.array(pos[-1]).round().astype(int)
    print('Read pixel: ', pos)

    print('Start reading video')
    clval = [] #np.expand_dims(frame[pos[1],pos[0]],0)
    while (ret):    
        clval.append(frame[pos[1],pos[0]])
        ret, frame = cap.read()
    
    cap.release()
    print('End reading video')
    
    digi = np.array(clval).sum(1)

    digi = digi>100
    digi = digi.astype(int)
    
    switch = np.diff(digi, axis=0)
    
    print('Start reading .xdf')
    stream = xdf.load_xdf(xdf_file, verbose = False)
    print('End reading .xdf')
    
    stream_names = np.array([item['info']['name'][0] for item in stream[0]])
    vid_sync = list(stream_names).index('Video Sync')
    
    vid_sync_times = np.array(stream[0][vid_sync]['time_stamps'])
    
    get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    record_frames = np.array(get_indices(1,switch))+1

    
    me,be = np.polyfit(record_frames[-2:], vid_sync_times[-2:], 1)
    ms, bs = np.polyfit(record_frames[:2], vid_sync_times[:2], 1)
    
    all_frames = np.arange(len(clval))
    frame_time = np.interp(all_frames, record_frames,vid_sync_times)
    frame_time[:record_frames[0]]= all_frames[:record_frames[0]]*ms+bs
    frame_time[record_frames[-1]+1:]=all_frames[record_frames[-1]+1:]*me+be
    
    return(frame_time)