import numpy as np


def get_temp_amplitude(peaks_p, peaks_m, filtered_arr, plot_times_zoomed_int, i):
    # Figure out the dT for each set of + and - peaks
    dT = []
    peak_times = []
    peak_times_ind = []
    mind = 0
    deltat = 0
    priordt = 1e5
    for pind in range(len(peaks_p)):
        done = False
        priordt = 1e6

        while(not done):
            deltat = np.abs(plot_times_zoomed_int[peaks_p[pind]] - plot_times_zoomed_int[peaks_m[mind]])/1e9
            if(deltat < 1200):
                dT.append(filtered_arr[peaks_p[pind], i] - filtered_arr[peaks_m[mind], i])
                peak_times.append(((plot_times_zoomed_int[peaks_p[pind]] + plot_times_zoomed_int[peaks_m[mind]])/2).astype('datetime64[ns]'))
                peak_times_ind.append(peaks_p[pind])
                # plt.plot([plot_times_zoomed_int[peaks_p[pind]].astype('datetime64[ns]'), plot_times_zoomed_int[peaks_m[mind]].astype('datetime64[ns]')], [filtered_arr[peaks_p[pind], i], filtered_arr[peaks_m[mind], i]], '--')
                if(mind < len(peaks_m) - 1):
                    mind = mind + 1
                break
            elif(priordt < deltat):
                break

            if(mind < len(peaks_m) - 1):
                mind = mind + 1
            else:
                break
            priordt = deltat
    return peak_times, dT, peak_times_ind