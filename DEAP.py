import scipy.io
import numpy as np
import scipy.signal as signal
import random

def preprocess_subject_dependent(filepath):
    mat = scipy.io.loadmat(filepath)
    data = mat['data']
    labels = mat['labels']
    labels = labels[:,[0,1]]
    ch_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2',
                'AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    rearranged_ch_indexs = [0,1,2,3,4,7,8,10,11,12,13,14,31,30,28,29,26,25,21,20,19,
                            17,16,18,5,6,9,15,27,24,22,23]
    data = data[0:32,:]
    sampling_rate = 128
    data= data[:,:,3*sampling_rate:]
    data = data[rearranged_ch_indexs,:,:]
    dt = data
    lb = labels
    random.seed(120)
    data_indices = list(range(data.shape[1]))
    np.random.shuffle(data_indices)
    data = data[:, data_indices, :]
    data_dict = {}
    labels_dict = {}
    for i in range(10):
        data_dict[str(i)] = data[:,i*4:(i+1)*4,:]
        labels_dict[str(i)] = labels[i*4:(i+1)*4,:]
        
    for key,dt in data_dict.items():
        lb = labels_dict[key]
        window_length = 3
        window_stride = 0.5
        segments = int((dt.shape[2]-window_length*sampling_rate)/(window_stride*sampling_rate) + 1)
        
        segmented_data_per_channel = np.zeros((dt.shape[0],dt.shape[1],segments, window_length*sampling_rate))
        segmented_trials = []
        segmented_data = []
        for i in range(dt.shape[0]):
            for j in range(dt.shape[1]):
                for k in range(segments):
                    start = k*window_stride
                    stop = window_length+start
                    segmented_data_per_channel[i,j,k,:] = dt[i,j,int(start*sampling_rate):int(stop*sampling_rate)]
        segmented_data_per_channel = np.array(segmented_data_per_channel)
        segmented_data_per_channel = np.transpose(segmented_data_per_channel,(0,3,2,1))
        vs = dt.shape[1]*115
        input_subject_dependent = np.zeros((32,384,vs))
        for i in range(segmented_data_per_channel.shape[0]):
            for j in range(segmented_data_per_channel.shape[1]):
                samples = []
                for k in range(segmented_data_per_channel.shape[2]):
                    samples.append(segmented_data_per_channel[i,j,k,:])
                    sample = np.array(samples)
                    sample = sample.reshape(-1)
                input_subject_dependent[i,j,:] = sample
        
        input_subject_dependent = input_subject_dependent.transpose((2,0,1))
        labels_segment = []
        for i in range(segmented_data_per_channel.shape[3]):
            for j in range(segmented_data_per_channel.shape[2]):
                labels_segment.append(lb[i,:])
        labels_segment = np.array(labels_segment)
        label_above_th = np.where(labels_segment > 5)
        label_below_th = np.where(labels_segment <= 5)
        labels_target = labels_segment.copy()
        labels_target[label_above_th] = 1
        labels_target[label_below_th] = 0
        labels_dict[key] = labels_target
        PhaseLockValue = []
        for s in range(input_subject_dependent.shape[0]):
            sample = input_subject_dependent[s,:,:]
            
            #Bandpass between 4-45
            sampling_freq = 128        # Sampling frequency (Hz)
            passband_freq = [4, 45]  # Passband frequencies (Hz)
            
            
            # Convert passband frequencies to normalized frequencies (0 to 1)
            nyquist_freq = 0.5 * sampling_freq
            low = passband_freq[0] / nyquist_freq
            high = passband_freq[1] / nyquist_freq
            
            # Design the bandpass IIR filter using Butterworth filter design
            n = 1001
            b = signal.firwin(n, cutoff = [low, high] , window = 'blackmanharris', pass_zero=False)
            
            for i in range(sample.shape[0]):
                sample[i,:] = signal.lfilter(b,1,sample[i,:])
            
            #Calculate PLV
            #Calculate the analytic signal
            plv_column = []
            for i in range(sample.shape[0]):
                    analytic_signal_1 = signal.hilbert(sample[i,:])
                    instantaneous_phase_1 = np.unwrap(np.angle(analytic_signal_1))
                    plv_row = []
                    for j in range(sample.shape[0]):
                        analytic_signal_2 = signal.hilbert(sample[j,:])
                        instantaneous_phase_2 = np.unwrap(np.angle(analytic_signal_2))
                        phase_difference = instantaneous_phase_1-instantaneous_phase_2
                        PLV = np.abs(np.mean(np.exp(1j * phase_difference)))
                        plv_row.append(PLV)
                    plv_column.append(plv_row)
            
            plv_column = np.array(plv_column)
            PhaseLockValue.append(plv_column)
        PhaseLockValue = np.array(PhaseLockValue)
        data_dict[key] = PhaseLockValue
    return data_dict,labels_dict




