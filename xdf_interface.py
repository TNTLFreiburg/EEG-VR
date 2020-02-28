import numpy as np
import xdf
import mne
from processors import StandardizeProcessor
from scipy.signal import filtfilt, iirnotch, butter


def xdf_loader(xdf_file):
    """ Loads an appropriate EEG file into a mne raw object.
    
    Args:
        xdf_file: The path to an .xdf file from which the data is extracted.
        
    Returns:
        raw: The mne raw object.
    
    """
    
    # Load the xdf file
    stream = xdf.load_xdf(xdf_file, verbose = False)
    
    # Extract the necessary event and eeg information
    stream_names = np.array([item['info']['name'][0] for item in stream[0]])
    game_state = list(stream_names).index('Game State')
    eeg_data = list(stream_names).index('NeuroneStream')

    sfreq = int(stream[0][eeg_data]['info']['nominal_srate'][0])
    game_state_series = np.array([item[0] for item in stream[0][game_state]['time_series']])
    game_state_times = np.array(stream[0][game_state]['time_stamps'])
    
    times = stream[0][eeg_data]['time_stamps']
    
    data = stream[0][eeg_data]['time_series'].T
    
    if (len(data) == 72):
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'emg', 'emg', 'emg', 'emg', 'eog', 'eog', 'eog', 'eog',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg']

        ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
                'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
                'Oz', 'O2', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'EMG_LF', 'EOG_R', 'EOG_L', 'EOG_U', 'EOG_D',
                'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
                'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
                'PO7', 'PO8']
        
    elif (len(data) == 75):
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'emg', 'emg', 'emg', 'emg', 'eog', 'eog', 'eog', 'eog',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'ecg', 'bio', 'bio']

        ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
                'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
                'Oz', 'O2', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'EMG_LF', 'EOG_R', 'EOG_L', 'EOG_U', 'EOG_D',
                'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
                'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
                'PO7', 'PO8', 'ECG', 'Respiration', 'GSR']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(data, info)
    
    events = np.zeros([game_state_series.size, 3], 'int')
    
    # Calculate the closest frame in the EEG data for each event time
    events[:,0] = [np.argmin(np.abs(times-event_time)) for event_time in game_state_times]
    
    legend = np.unique(game_state_series)
    class_vector = np.zeros(game_state_series.size, dtype='int')
    event_id = {}
    for ii in np.arange(legend.size):
        class_vector += (game_state_series == legend[ii])*ii
        event_id[legend[ii]] = ii
        
    events[:,2] = class_vector
    
    raw.events = events
    raw.event_id = event_id

    # Set eeg sensor locations
    raw.set_montage(mne.channels.read_montage("standard_1005", ch_names=raw.info["ch_names"]))

    return(raw)

def bdonline_extract(path, files, timeframe_start, target_fps, emg = false):
    """ Uses event markers to extract motor tasks from multiple DLVR .xdf files.
    Args:
        path: If the files share a single path, you can specify it here.
        
        files: A list of .xdf files to extract data from.
        
        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.
        
        target_fps: Downsample the EEG-data to this value.         
    
    Returns:
        X: A list of trials
        y: An array specifying the action
    """
    
    DOWNSAMPLING_COEF = int(5000 / target_fps)
    processor = StandardizeProcessor()
    # Epochs list containing differently sized arrays [#eeg_electrodes, times]
    X = []
    #event ids corresponding to the trials where 'left' = 0 and 'right' = 1
    y = np.array([])
    for file in files:
        #load a file
        print('Reading ', file)
        current_raw = xdf_loader(path+file)
        
        if emg:
            current_raw._data[14,:]= current_raw.get_data(picks = ['EMG_LF']) #C3
            current_raw._data[16,:]= current_raw.get_data(picks = ['EMG_RF']) #C4
        
            current_raw._data[55,:]= current_raw.get_data(picks = ['EMG_LF']) #CP3
            current_raw._data[57,:]= current_raw.get_data(picks = ['EMG_RF']) #CP4
        
        #discard EOG/EMG
        current_raw.pick_types(meg=False, eeg=True)
        #pick only relevant events
        events = current_raw.events[(current_raw.events[:,2]==current_raw.event_id['Monster left']) | (current_raw.events[:,2]==current_raw.event_id['Monster right'])]
        #timestamps where a monster deactivates
        stops = current_raw.events[:,0][(current_raw.events[:,2]==current_raw.event_id['Monster destroyed'])]
        #timestamps where trials begin
        starts = events[:,0]
    
        #extract event_ids and shift them to [0, 1] -> 'Monster left' = 0 / 'Monster right' = 1
        key = events[:,2]
        key = (key==key.max()).astype(np.int64)

        # Butter filter (lowpass) for 40 Hz (EEG) or 120 Hz (EMG)
        if emg:
            cf = 120
        else: 
            cf = 40
        B_40, A_40 = butter(6, cf, btype='low', output='ba', fs = 5000)

        # Notch filter with 50 HZ
        F0 = 50.0
        Q = 30.0  # Quality factor
        # Design notch filter
        B_50, A_50 = iirnotch(F0, Q, 5000)
        
        bads = np.array([])
        for count, event in enumerate(starts):
            #in case the last trial has no end (experiment ended before the trial ends), discard it
            if len(stops[stops>event]) == 0:
                key = key[:-sum(starts>=event)]
                break
                
            if stops[stops>event][0]-event < 5000:
                bads = np.append(bads,count)                
                continue
            
            #Get the trial from 1 second before the task starts to the next 'Monster deactived' flag
            current_epoch = np.array([]) #current_raw._data[:, event-round(timeframe_start*5000) : stops[stops>event][0]]
            #400 samples each
            ii = event-round(timeframe_start*5000)
            while ii < stops[stops>event][0]:
                current_chunk = current_raw._data[:,ii:ii+400] 
                ii += 400
        
                current_chunk = filtfilt(B_50, A_50, current_chunk)
                current_chunk = filtfilt(B_40, A_40, current_chunk)            
            
                #downsample to 250 Hz
                current_chunk = np.array([np.mean(current_chunk[:, i:i + DOWNSAMPLING_COEF], axis=1) for i in
							np.arange(0, 400, DOWNSAMPLING_COEF)]).astype('float32')
                
                current_chunk = processor.process(current_chunk).T
                
                current_chunk = current_chunk.astype(np.float32)
                
                if current_epoch.size == 0:
                    current_epoch = current_chunk                  
                else:
                    current_epoch = np.append(current_epoch, current_chunk, 1)  
                    
            
            X.append(current_epoch)
        
        if len(bads)>0:
            key = np.delete(key, bads)
        y = np.append(y,key)
    y = y.astype(np.int64)
    return(X,y)