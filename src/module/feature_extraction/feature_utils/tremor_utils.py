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
    f_values, psd_values = welch(y_values, fs)
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

def mAmp(data):  #
    L = np.size(data, 0)
    upper_envolope, low_envolope = envelope_extraction(data)
    mAmp = np.sum(upper_envolope-low_envolope)/L*1.0
    return mAmp


def envelope_extraction(signal):
    s = signal.astype(float )
    q_u = np.zeros(s.shape)
    q_l =  np.zeros(s.shape)

    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [s[0],]

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


    # u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # for k in range(0,len(s)):
    #   q_u[k] = u_p(k)
    #   q_l[k] = l_p(k)

    return upper_envelope_y, lower_envelope_y

def general_equation(first_x, first_y, second_x, second_y):

    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b


def fft_peak_xy(data,N,fs,peak_num=2):
    f_values, fft_values = get_fft_values(data, N, fs)
    signal_min = np.nanpercentile(fft_values, 5)
    signal_max = np.nanpercentile(fft_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(fft_values)  # set minimum peak height
    peaks, _ = find_peaks(fft_values, prominence=mph)
    peak_save = fft_values[peaks].argsort()[::-1][:peak_num]
    temp_arr = f_values[peaks[peak_save]]
    fft_peak_xy = np.pad(temp_arr, (0, peak_num - len(temp_arr)), 'constant', constant_values=0)
    return fft_peak_xy


def psd_peak_xy(data,N,fs,peak_num=2):
    p_values, psd_values = get_psd_values(data, N, fs)
    signal_min = np.nanpercentile(psd_values, 5)
    signal_max = np.nanpercentile(psd_values, 95)
    mph = signal_min + (signal_max - signal_min) / len(psd_values)  # set minimum peak height
    peaks3, _ = find_peaks(psd_values, height=mph)
    peak_save = psd_values[peaks3].argsort()[::-1][:peak_num]
    temp_arr = psd_values[peaks3[peak_save]]
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
    # 偏度
    seriesdata = pd.Series(data)
    skew = seriesdata.skew()
    # 峰度
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
    # a = pd.value_counts(data) / len(data)
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
    # print("A", A)
    # print("B", B)
    # Return SampEn
    return -np.log(A / B)