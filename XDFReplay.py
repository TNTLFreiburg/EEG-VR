import numpy as np
import xdf
import pylsl as lsl
import time
import keyboard
from scipy import signal


def replay_xdf(path, files, auto = True):
    game_info = lsl.StreamInfo("Game State", "Flags", 1, 0, 'string', "jkllkjl")
    game_outlet =  lsl.StreamOutlet(game_info)

    eeg_info = lsl.StreamInfo("NeuroneStream", "EEG", 75, 5000, 'float32', "SDQWD")
    eeg_outlet = lsl.StreamOutlet(eeg_info)
    
    for file in files:
        #load a file
        print('Reading ', file)
        stream = xdf.load_xdf(path+file, verbose = False)
        
        stream_names = np.array([item['info']['name'][0] for item in stream[0]])
        game_state = list(stream_names).index('Game State')
        eeg_data = list(stream_names).index('NeuroneStream')
        
        game_series = stream[0][game_state]['time_series']
        game_time = stream[0][game_state]['time_stamps']
        
        eeg_series = stream[0][eeg_data]['time_series']
        eeg_time = stream[0][eeg_data]['time_stamps']
        
        eeg_time = eeg_time - game_time[0]
        game_time = game_time - game_time[0]
        
        game_idx = 0
        eeg_idx = eeg_time[eeg_time<=0].size-1
        end = game_time.size-1
        
        if not auto:
            print('Press "Enter" to start streaming next file.')
            keyboard.wait('enter')
        
        start = time.time()
        min = 0
        now = time.time()-start

        while game_idx <= end:
            now = time.time()-start
            game_target = game_time[game_time<=now].size-1
            eeg_target = eeg_time[eeg_time<=now].size-1
            while game_idx <= game_target:
                game_outlet.push_sample(game_series[game_idx])
                game_idx += 1
            while eeg_idx <= eeg_target:
                eeg_outlet.push_sample(eeg_series[eeg_idx])
                eeg_idx += 1
            if now / 60 > min:
                print("Streamed for",min,"out of 5 minutes.")
                min += 1
    print("Streaming complete.")