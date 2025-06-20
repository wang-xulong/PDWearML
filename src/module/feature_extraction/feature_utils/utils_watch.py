from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import butter, lfilter
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft,fftshift
from scipy.special import entr
import pandas as pd
from scipy.integrate import simpson
import numpy as np
from numpy import array, sign, zeros

def get_psd_values(y_values, N, fs):
    a = len(y_values)
    f_values, psd_values = welch(y_values, fs, nperseg=150)
    return f_values, psd_values

def get_fft_values(y_values, N, fs):
    f_values = np.linspace(0.0, fs / 2.0, N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]

def get_autocorr_values(y_values, N, fs):
    autocorr_values = autocorr(y_values)
    x_values = np.array([1.0 * jj / fs for jj in range(0, N)])
    return x_values, autocorr_values

def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a

def butter_highpass_filter(data, cutOff, fs, order=3):
    b, a = butter_highpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def fft_peak_xy(data,N,fs,peak_num=2):
    f_values, fft_values = get_fft_values(data, N, fs)
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:peak_num]
    temp_arr = f_values[peaks[peak_save]] + fft_values[peaks[peak_save]]
    fft_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant', constant_values=0)
    return fft_peak_xy

def psd_peak_xy(data,N,fs,peak_num=2):
    p_values, psd_values = get_psd_values(data, N, fs)
    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks3, _ = find_peaks(psd_values, height=mph)
    peak_save = psd_values[peaks3].argsort()[::-1][:peak_num]
    temp_arr = p_values[peaks3[peak_save]] + psd_values[peaks3[peak_save]]
    psd_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                              constant_values=0)
    return psd_peak_xy


def auto_peak_xy(data,N,fs,peak_num=2):
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks4, _ = find_peaks(autocorr_values, height=mph)
    peak_save = autocorr_values[peaks4].argsort()[::-1][:peak_num]
    temp_arr = a_values[peaks4[peak_save]] + autocorr_values[peaks4[peak_save]]
    autocorr_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant',
                                   constant_values=0)
    return autocorr_peak_xy


def envelope_extraction(signal):
    s = signal.astype(float )
    q_u = np.zeros(s.shape)
    q_l =  np.zeros(s.shape)


    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [s[0],] #

    l_x = [0,]
    l_y = [s[0],]


    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1,len(s)-1):
        if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    l_x.append(len(s) - 1)
    l_y.append(s[-1])


    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]
    lower_envelope_y[-1] = l_y[-1]


    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1

            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])


    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1

            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])


    return upper_envelope_y, lower_envelope_y

def general_equation(first_x, first_y, second_x, second_y):
    # 斜截式 y = kx + b
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def mAmp(data):
    L = np.size(data, 0)
    upper_envolope, low_envolope = envelope_extraction(data)
    mAmp = np.sum(upper_envolope-low_envolope)/L*1.0
    return mAmp

def base(data):
    damp = mAmp(data)
    dmean = data.mean()
    dmax = data.max()
    dstd = data.std()
    dvar = data.var()
    # entropy
    dentr = entr(abs(data)).sum(axis=0) / np.log(10)
    # log_energy
    log_energy_value = np.log10(data ** 2).sum(axis=0)
    # SMA
    time = np.arange(data.shape[0])
    signal_magnitude_area = simpson(data,time)
    # Interquartile range (interq)
    per25 = np.nanpercentile(data, 25)
    per75 = np.nanpercentile(data, 75)
    interq = per75 - per25

    seriesdata = pd.Series(data)
    skew = seriesdata.skew()

    kurt = seriesdata.kurt()
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt

def time_domain(data):
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(data)
    drms = np.sqrt((np.square(data).mean()))  # rms
    return  damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms

def corrcoef(x,y,z,a):
    xy_cor = np.corrcoef(x, y)
    xz_cor = np.corrcoef(x, z)
    xa_cor = np.corrcoef(x, a)
    yz_cor = np.corrcoef(y, z)
    ya_cor = np.corrcoef(y, a)
    za_cor = np.corrcoef(z, a)
    return xy_cor[0, 1], xz_cor[0, 1], xa_cor[0, 1], yz_cor[0, 1], ya_cor[0, 1], za_cor[0, 1]

def fft_domain(data,N,fs):
    f_values, fft_values = get_fft_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(fft_values)
    drms = np.sqrt((np.square(fft_values).mean()))  # rms
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms


def psd_domain(data,N,fs):
    p_values, psd_values = get_psd_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(psd_values)
    drms = np.sqrt((np.square(psd_values).mean()))  # rms
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms


def autocorr_domain(data,N,fs):
    a_values, autocorr_values = get_autocorr_values(data, N, fs)
    damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt = base(autocorr_values)
    drms = np.sqrt((np.square(autocorr_values).mean()))  # rms

    signal_min = np.nanpercentile(autocorr_values, 5)
    signal_max = np.nanpercentile(autocorr_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(autocorr_values)  # set minimum peak height
    peaks, _ = find_peaks(autocorr_values, prominence=mph)

    peak_save = autocorr_values[peaks].argsort()[::-1][:2]
    peak_x = a_values[peaks[peak_save]]
    peak_y = autocorr_values[peaks[peak_save]]
    peak_x = np.pad(peak_x, (0, 2 - len(peak_x)), 'constant', constant_values=0)
    peak_y = np.pad(peak_y, (0, 2 - len(peak_y)), 'constant', constant_values=0)

    peak_main_Y = peak_y[0]

    peak_sub_Y = peak_y[1]

    cftor = peak_main_Y / drms * 1.0
    return damp, dmean, dmax, dstd, dvar, dentr, log_energy_value, signal_magnitude_area, interq, skew, kurt, drms,  peak_main_Y,  peak_sub_Y, cftor

def get_psd_values(y_values, N, fs):
    f_values, psd_values = welch(y_values, fs)
    return f_values, psd_values

def infor(data):
    a = pd.Series(data).value_counts() / len(data)
    return sum(np.log2(a) * a * (-1))

def get_fft_values(y_values, N, fs):
    f_values = np.linspace(0.0, fs / 2.0, N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, N, fs):
    autocorr_values = autocorr(y_values)
    x_values = np.array([1.0 * jj / fs for jj in range(0, N)])
    return x_values, autocorr_values


def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a


def butter_highpass_filter(data, cutOff, fs, order=3):
    b, a = butter_highpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sampEn(data, N, r):
    L = len(data)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([data[i: i + N] for i in range(L - N)])
    xmj = np.array([data[i: i + N] for i in range(L - N + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    N += 1
    xm = np.array([data[i: i + N] for i in range(L - N + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    # Return SampEn
    # print(A, B)
    return -np.log(A / B)

def featureExtract(x, y, z, ACCW2, windowsize, overlapping, frequency):
    N = windowsize
    fs = frequency
    f_s = frequency
    data = ACCW2
    i = 0
    j = 0

    # xmean = x.mean()
    # xvar = x.var()
    # ymean = y.mean()
    # yvar = y.var()
    # zmean = z.mean()
    # zvar = z.var()
    # amean = ACCW2.mean()
    # avar = ACCW2.var()

    # z_filter = tremor_utils.butter_bandpass_filter(z,0.2, 2, 200, order = 4)
    z_filter = np.array(z).flatten()
    signal_min = np.nanpercentile(z_filter, 5)
    signal_max = np.nanpercentile(z_filter, 97)
    mph = signal_max + (signal_max - signal_min) / len(z_filter)  # set minimum peak height
    peaks_t, _ = find_peaks(z_filter, prominence=mph, distance=120)
    peak_num = len(peaks_t)  ##z轴peak数量
    t_value = np.arange(len(z_filter))
    t_peakmax = np.argsort(z_filter[peaks_t])[-1]

    infory = infor(t_value[peaks_t])  #
    inforx = infor(z_filter[peaks_t])  #

    # t_peakmax_X = t_value[peaks_t[t_peakmax]]
    t_peakmax_Y = z_filter[peaks_t[t_peakmax]]
    t_peak_y = z_filter[peaks_t]
    dyski_num = len(t_peak_y[(t_peak_y < t_peakmax_Y - mph)])

    # auto_X = a_values[peaks4[index_peakmax]]
    a_values, autocorr_values = get_autocorr_values(z_filter, N, fs)
    peaks4, _ = find_peaks(autocorr_values)
    auto_peak_num = len(peaks4)
    index_peakmax = np.argsort(autocorr_values[peaks4])[-1]
    # print(autocorr_values[peaks4])
    auto_y = autocorr_values[peaks4[index_peakmax]]

    # whole
    peaks_normal = np.zeros([len(data), 1], dtype=float)
    peaks_abnormal = np.zeros([len(data), 1], dtype=float)
    fea_autoy = np.zeros([len(data), 1], dtype=float)
    fea_auto_num = np.zeros([len(data), 1], dtype=float)
    # time domain 12
    time_domain1 = np.zeros([len(data), 12], dtype=float)
    time_domain2 = np.zeros([len(data), 12], dtype=float)
    time_domain3 = np.zeros([len(data), 12], dtype=float)
    time_domain4 = np.zeros([len(data), 12], dtype=float)
    time_axiscof = np.zeros([len(data), 6], dtype=float)
    # frequency domain 12
    fre_domain1 = np.zeros([len(data), 12], dtype=float)
    fre_domain2 = np.zeros([len(data), 12], dtype=float)
    fre_domain3 = np.zeros([len(data), 12], dtype=float)
    fre_domain4 = np.zeros([len(data), 12], dtype=float)
    fft_peak_a = np.zeros([len(data), 2], dtype=float)
    # psd domain 12
    psd_domain1 = np.zeros([len(data), 12], dtype=float)
    psd_domain2 = np.zeros([len(data), 12], dtype=float)
    psd_domain3 = np.zeros([len(data), 12], dtype=float)
    psd_domain4 = np.zeros([len(data), 12], dtype=float)
    psd_peak_a = np.zeros([len(data), 2], dtype=float)
    # auto domain 15
    autoco_domain1 = np.zeros([len(data), 15], dtype=float)
    autoco_domain2 = np.zeros([len(data), 15], dtype=float)
    autoco_domain3 = np.zeros([len(data), 15], dtype=float)
    autoco_domain4 = np.zeros([len(data), 15], dtype=float)
    autocorr_peak_a = np.zeros([len(data), 2], dtype=float)

    # print("len_data-windowsize", len(data) - windowsize)

    while (i < int(len(data) - windowsize)):
        data1 = x[int(i):int(i + windowsize)]
        data2 = y[int(i):int(i + windowsize)]
        data3 = z[int(i):int(i + windowsize)]
        data4 = ACCW2[int(i):int(i + windowsize)]
        data1 = data1.values
        data2 = data2.values
        data3 = data3.values
        data4 = data4.values
        # ***************************************
        peaks_normal[j, :] = peak_num  #
        peaks_abnormal[j, :] = dyski_num  #

        fea_autoy[j, :] = auto_y
        fea_auto_num[j, :] = auto_peak_num

        # ***************************************short term features******************************
        time_domain1[j, :] = time_domain(data1)  # 14
        time_domain2[j, :] = time_domain(data2)
        time_domain3[j, :] = time_domain(data3)
        time_domain4[j, :] = time_domain(data4)
        time_axiscof[j, :] = corrcoef(data1, data2, data3, data4)
        fre_domain1[j, :] = fft_domain(data1, N, fs)  # 19
        fre_domain2[j, :] = fft_domain(data2, N, fs)
        fre_domain3[j, :] = fft_domain(data3, N, fs)
        fre_domain4[j, :] = fft_domain(data4, N, fs)
        fft_peak_a[j, :] = fft_peak_xy(data4, N, fs, peak_num=2)
        psd_domain1[j, :] = psd_domain(data1, N, fs)  # 19
        psd_domain2[j, :] = psd_domain(data2, N, fs)
        psd_domain3[j, :] = psd_domain(data3, N, fs)
        psd_domain4[j, :] = psd_domain(data4, N, fs)
        psd_peak_a[j, :] = psd_peak_xy(data4, N, fs, peak_num=2)

        data1234 = np.c_[data1, data2, data3, data4]
        data1234 = StandardScaler().fit_transform(data1234)  # 19
        data1 = data1234[:, 0]
        data2 = data1234[:, 1]
        data3 = data1234[:, 2]
        data4 = data1234[:, 3]
        autoco_domain1[j, :] = autocorr_domain(data1, N, fs)
        autoco_domain2[j, :] = autocorr_domain(data2, N, fs)
        autoco_domain3[j, :] = autocorr_domain(data3, N, fs)
        autoco_domain4[j, :] = autocorr_domain(data4, N, fs)
        autocorr_peak_a[j, :] = auto_peak_xy(data4, N, fs, peak_num=2)

        i = i + windowsize * (1 - overlapping) - 1
        j = j + 1


    fea_whole = np.c_[peaks_normal, peaks_abnormal, fea_autoy, fea_auto_num]
    f1 = np.c_[time_axiscof, fft_peak_a, psd_peak_a, autocorr_peak_a]
    # 20，25，26，24
    fx = np.c_[time_domain1, fre_domain1, psd_domain1, autoco_domain1]
    fy = np.c_[time_domain2, fre_domain2, psd_domain2, autoco_domain2]
    fz = np.c_[time_domain3, fre_domain3, psd_domain3, autoco_domain3]
    fa = np.c_[time_domain4, fre_domain4, psd_domain4, autoco_domain4]

    Feat = np.c_[fea_whole, f1, fx, fy, fz, fa]

    Feat2 = np.zeros((j, Feat.shape[1]))  #
    Feat2[0:j, :] = Feat[0:j, :]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2


def FeatureExtractWithProcess1(patients_id, activity_id, sensor, data_path, side, fea_column, window_size, overlapping_rate,
                              frequency):
    Feature = pd.DataFrame()
    for pdn in patients_id:
        for acn in activity_id:
            # select one side data
            filefullpath = data_path + "person{}/{}_session{}_{}.csv".format(pdn, pdn, acn, side)
            if not os.path.exists(filefullpath):
                continue
            data = pd.read_csv(filefullpath, header=0)
            data = data.drop(0)

            new_column_labels = {"Acc_x_100": "wr_acc_x", "Acc_y_100": "wr_acc_y",
                                 "Acc_z_100": "wr_acc_z", "Gyro_x_100": "gyro_x", "Gyro_y_100": "gyro_y",
                                 "Gyro_z_100": "gyro_z"}

            data = data.rename(columns=new_column_labels)
            # 解决acc和gyro长度不一致的问题，只是简单的去除空值
            data = data.replace([np.inf, -np.inf], np.nan).dropna()

            if(sensor == "acc"):
                for col in ["wr_acc_x", "wr_acc_y", "wr_acc_z"]:
                    data[col] = data[col].astype('float64')
                data["acca"] = np.sqrt(
                    data["wr_acc_x"] * data["wr_acc_x"] + data["wr_acc_y"] * data["wr_acc_y"] + data["wr_acc_z"] * data[
                        "wr_acc_z"])
                accdata = data[["wr_acc_x", "wr_acc_y", "wr_acc_z", "acca"]]
                accdata = accdata.values
                # 输入需要为numpy数组
                accdata = StandardScaler().fit_transform(accdata)
                databand_acc = accdata.copy()
                for k in range(0, 4):
                    databand_acc[:, k] = butter_bandpass_filter(accdata[:, k], 0.3, 17, 100, order=3)
                #databand_acc[:, 2] = butter_bandpass_filter(accdata[:, 2], 0.3, 3, 100, order=3)
                databand_acc = pd.DataFrame(databand_acc)
                databand = databand_acc
            elif(sensor == "gyro"):
                for col in ["gyro_x", "gyro_y", "gyro_z"]:
                    data[col] = data[col].astype('float64')
                data["gyroa"] = np.sqrt(
                    data["gyro_x"] * data["gyro_x"] + data["gyro_y"] * data["gyro_y"] + data["gyro_z"] * data["gyro_z"])
                gyrodata = data[["gyro_x", "gyro_y", "gyro_z", "gyroa"]]
                gyrodata = gyrodata.values
                gyrodata = StandardScaler().fit_transform(gyrodata)
                databand_gyro = gyrodata.copy()
                for k in range(0, 4):
                    databand_gyro[:, k] = butter_bandpass_filter(gyrodata[:, k], 0.3, 17, 100, order=3)
                databand_gyro[:, 2] = butter_bandpass_filter(gyrodata[:, 2], 0.3, 3, 100, order=3)
                databand_gyro = pd.DataFrame(databand_gyro)
                databand = databand_gyro

            datax = databand.iloc[:, 0]
            datay = databand.iloc[:, 1]
            dataz = databand.iloc[:, 2]
            acca = databand.iloc[:, 3]

            feature = featureExtract(datax, datay, dataz, acca, window_size, overlapping_rate, frequency)
            feature.columns = fea_column
            feature["PatientID"] = pdn
            feature["activity_label"] = acn
            print(f"End of processing patients {pdn}, activity {acn}")
            if ((pdn == 1) & (acn == 1)):
                Feature = feature
            else:
                Feature = pd.concat([Feature, feature], axis=0)

    return Feature