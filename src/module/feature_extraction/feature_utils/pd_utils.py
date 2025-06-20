from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import *
from sklearn.preprocessing import StandardScaler
import os

from .import tremor_utils



def featureExtract(x, y, z, ACCW2, windowsize, overlapping, frequency):
    N = windowsize
    fs = frequency
    data = ACCW2
    i = 0
    j = 0

    # z_filter = tremor_utils.butter_bandpass_filter(z,0.2, 2, 200, order = 4)
    z_filter = np.array(z).flatten()
    signal_min = np.nanpercentile(z_filter, 5)
    signal_max = np.nanpercentile(z_filter, 97)
    mph = signal_max + (signal_max - signal_min) / len(z_filter)  # set minimum peak height
    peaks_t, _ = find_peaks(z_filter, prominence=mph, distance=120)
    peak_num = len(peaks_t)
    t_value = np.arange(len(z_filter))
    t_peakmax = np.argsort(z_filter[peaks_t])[-1]


    # t_peakmax_X = t_value[peaks_t[t_peakmax]]
    t_peakmax_Y = z_filter[peaks_t[t_peakmax]]  #
    t_peak_y = z_filter[peaks_t]
    dyski_num = len(t_peak_y[(t_peak_y < t_peakmax_Y - mph)])

    # auto_X = a_values[peaks4[index_peakmax]]
    a_values, autocorr_values = tremor_utils.get_autocorr_values(z_filter, N, fs)
    peaks4, _ = find_peaks(autocorr_values)
    auto_peak_num = len(peaks4)
    if len(peaks4) > 0:
        index_peakmax = np.argsort(autocorr_values[peaks4])[-1]
        auto_y = autocorr_values[peaks4[index_peakmax]]
    else:
        index_peakmax = 0
        auto_y = 0


    # whole
    peak_num1 = 2
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
    fft_peak_a = np.zeros([len(data), peak_num1], dtype=float)
    # psd domain 12
    psd_domain1 = np.zeros([len(data), 12], dtype=float)
    psd_domain2 = np.zeros([len(data), 12], dtype=float)
    psd_domain3 = np.zeros([len(data), 12], dtype=float)
    psd_domain4 = np.zeros([len(data), 12], dtype=float)
    psd_peak_a = np.zeros([len(data), peak_num1], dtype=float)
    # auto domain 15
    autoco_domain1 = np.zeros([len(data), 15], dtype=float)
    autoco_domain2 = np.zeros([len(data), 15], dtype=float)
    autoco_domain3 = np.zeros([len(data), 15], dtype=float)
    autoco_domain4 = np.zeros([len(data), 15], dtype=float)
    autocorr_peak_a = np.zeros([len(data), peak_num1], dtype=float)

    while (i < len(data) - windowsize):
        data1 = x[i:i + windowsize]
        data2 = y[i:i + windowsize]
        data3 = z[i:i + windowsize]
        data4 = ACCW2[i:i + windowsize]
        data1 = data1.values
        data2 = data2.values
        data3 = data3.values
        data4 = data4.values

        peaks_normal[j, :] = peak_num
        peaks_abnormal[j, :] = dyski_num

        fea_autoy[j, :] = auto_y
        fea_auto_num[j, :] = auto_peak_num

        # ***************************************short term features******************************
        time_domain1[j, :] = tremor_utils.time_domain(data1)  # 14
        time_domain2[j, :] = tremor_utils.time_domain(data2)
        time_domain3[j, :] = tremor_utils.time_domain(data3)
        time_domain4[j, :] = tremor_utils.time_domain(data4)
        time_axiscof[j, :] = tremor_utils.corrcoef(data1, data2, data3, data4)
        fre_domain1[j, :] = tremor_utils.fft_domain(data1, N, fs)  # 19
        fre_domain2[j, :] = tremor_utils.fft_domain(data2, N, fs)
        fre_domain3[j, :] = tremor_utils.fft_domain(data3, N, fs)
        fre_domain4[j, :] = tremor_utils.fft_domain(data4, N, fs)
        fft_peak_a[j, :] = tremor_utils.fft_peak_xy(data4, N, fs, peak_num1)
        psd_domain1[j, :] = tremor_utils.psd_domain(data1, N, fs)  # 19
        psd_domain2[j, :] = tremor_utils.psd_domain(data2, N, fs)
        psd_domain3[j, :] = tremor_utils.psd_domain(data3, N, fs)
        psd_domain4[j, :] = tremor_utils.psd_domain(data4, N, fs)
        psd_peak_a[j, :] = tremor_utils.psd_peak_xy(data4, N, fs, peak_num1)

        data1234 = np.c_[data1, data2, data3, data4]
        data1234 = StandardScaler().fit_transform(data1234)  # 19
        data1 = data1234[:, 0]
        data2 = data1234[:, 1]
        data3 = data1234[:, 2]
        data4 = data1234[:, 3]
        autoco_domain1[j, :] = tremor_utils.autocorr_domain(data1, N, fs)
        autoco_domain2[j, :] = tremor_utils.autocorr_domain(data2, N, fs)
        autoco_domain3[j, :] = tremor_utils.autocorr_domain(data3, N, fs)
        autoco_domain4[j, :] = tremor_utils.autocorr_domain(data4, N, fs)
        autocorr_peak_a[j, :] = tremor_utils.auto_peak_xy(data4, N, fs, peak_num1)

        assert 0 <= overlapping < 1
        i = int(i + windowsize * (1-overlapping) - 1)
        j = j + 1

    # whole特征
    fea_whole = np.c_[peaks_normal, peaks_abnormal, fea_autoy, fea_auto_num]
    f1 = np.c_[time_axiscof, fft_peak_a, psd_peak_a, autocorr_peak_a]
    # 20，25，26，24
    fx = np.c_[time_domain1, fre_domain1, psd_domain1, autoco_domain1]
    fy = np.c_[time_domain2, fre_domain2, psd_domain2, autoco_domain2]
    fz = np.c_[time_domain3, fre_domain3, psd_domain3, autoco_domain3]
    fa = np.c_[time_domain4, fre_domain4, psd_domain4, autoco_domain4]

    Feat = np.c_[fea_whole, f1, fx, fy, fz, fa]

    Feat2 = np.zeros((j, Feat.shape[1]))
    Feat2[0:j, :] = Feat[0:j, :]
    Feat2 = pd.DataFrame(Feat2)
    return Feat2

def FeatureExtractWithProcess(patients_id, activity_id, data_path, side, fea_column, window_size, overlapping_rate,
                              frequency):
    Feature = pd.DataFrame()
    for pdn in patients_id:
        for acn in activity_id:
            # select one side data
            filefullpath = data_path + r"person{}/{}_session{}_{}.csv".format(pdn, pdn, acn, side)
            if not os.path.exists(filefullpath):
                continue
            data = pd.read_csv(filefullpath, header=0)
            if 'Acc_x_100' not in data.columns:
                data = data.drop(0)

            # new_column_labels = {"Accel_WR_X_CAL": "wr_acc_x", "Accel_WR_Y_CAL": "wr_acc_y",
            #                      "Accel_WR_Z_CAL": "wr_acc_z", "Gyro_X_CAL": "gyro_x", "Gyro_Y_CAL": "gyro_y",
            #                      "Gyro_Z_CAL": "gyro_z"}
            new_column_labels = {# shimmer
                                 "Accel_WR_X_CAL": "wr_acc_x",
                                 "Accel_WR_Y_CAL": "wr_acc_y",
                                 "Accel_WR_Z_CAL": "wr_acc_z",
                                 "Gyro_X_CAL": "gyro_x",
                                 "Gyro_Y_CAL": "gyro_y",
                                 "Gyro_Z_CAL": "gyro_z",
                                 "Mag_X_CAL": "mag_x",
                                 "Mag_Y_CAL": "mag_y",
                                 "Mag_Z_CAL": "mag_z",
                                # watch
                                 "Acc_x_100": "wr_acc_x",
                                 "Acc_y_100": "wr_acc_y",
                                 "Acc_z_100": "wr_acc_z",
                                 "Gyro_x_100": "gyro_x",
                                 "Gyro_y_100": "gyro_y",
                                 "Gyro_z_100": "gyro_z",
                                 }
            data = data.rename(columns=new_column_labels)

            if all(col in data.columns for col in ["wr_acc_x", "wr_acc_y", "wr_acc_z"]):
                for col in ["wr_acc_x", "wr_acc_y", "wr_acc_z"]:
                    # data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].astype('float64')
            if all(col in data.columns for col in ["gyro_x", "gyro_y", "gyro_z"]):
                for col in ["gyro_x", "gyro_y", "gyro_z"]:
                    # data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].astype('float64')
            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                for col in ["mag_x", "mag_y", "mag_z"]:
                    data[col] = data[col].astype('float64')

            data["acca"] = np.sqrt(
                data["wr_acc_x"] * data["wr_acc_x"] + data["wr_acc_y"] * data["wr_acc_y"] + data["wr_acc_z"] * data[
                    "wr_acc_z"])
            data["gyroa"] = np.sqrt(
                data["gyro_x"] * data["gyro_x"] + data["gyro_y"] * data["gyro_y"] + data["gyro_z"] * data["gyro_z"])

            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                data["maga"] = np.sqrt(
                    data["mag_x"] * data["mag_x"] + data["mag_y"] * data["mag_y"] + data["mag_z"] * data["mag_z"])

            accdata = data[["wr_acc_x", "wr_acc_y", "wr_acc_z", "acca"]]
            gyrodata = data[["gyro_x", "gyro_y", "gyro_z", "gyroa"]]

            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                magdata = data[["mag_x", "mag_y", "mag_z", "maga"]]

            accdata = accdata.values
            gyrodata = gyrodata.values

            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                magdata = magdata.values

            accdata = StandardScaler().fit_transform(accdata)  # z-score
            gyrodata = StandardScaler().fit_transform(gyrodata)
            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                magdata = StandardScaler().fit_transform(magdata)
            databand_acc = accdata.copy()
            databand_gyro = gyrodata.copy()

            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                databand_mag = magdata.copy()

            for k in range(0, 4):
                databand_acc[:, k] = tremor_utils.butter_bandpass_filter(accdata[:, k], 0.3, 17, 200, order=3)   # 滤波
                databand_gyro[:, k] = tremor_utils.butter_bandpass_filter(gyrodata[:, k], 0.3, 17, 200, order=3)
                if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):# 滤波
                    databand_mag[:, k] = tremor_utils.butter_bandpass_filter(magdata[:, k], 0.3, 17, 200, order=3)  # 滤波

            databand_acc = pd.DataFrame(databand_acc)
            databand_gyro = pd.DataFrame(databand_gyro)

            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                databand_mag = pd.DataFrame(databand_mag)

            # # =================================================================
            feature_ls = []
            if all(col in data.columns for col in ["mag_x", "mag_y", "mag_z"]):
                databand_list = [databand_acc, databand_gyro, databand_mag]
            else:
                databand_list = [databand_acc, databand_gyro]
            for databand in databand_list:
                datax = databand.iloc[:, 0]
                datay = databand.iloc[:, 1]
                dataz = databand.iloc[:, 2]
                dataa = databand.iloc[:, 3]
                feature_ls.append(featureExtract(datax, datay, dataz, dataa, window_size, overlapping_rate, frequency))
            feature = pd.concat(feature_ls, axis=1)
            feature.columns = fea_column

            feature["PatientID"] = pdn
            feature["activity_label"] = acn
            print(f"End of processing patients {pdn}, activity {acn}")
            if ((pdn == 1) & (acn == 1)):
                Feature = feature
            else:
                Feature = pd.concat([Feature, feature], axis=0)

    return Feature