import numpy as np
import scipy
from tqdm import tqdm
from PyAstronomy import pyaC

def DTS_to_disp(n_contours, filtered_arr, plot_times_zoomed, plot_dists_zoomed):
    contour_values = np.linspace(np.percentile(filtered_arr[~np.isnan(filtered_arr)],1), np.percentile(filtered_arr[~np.isnan(filtered_arr)],99), n_contours) # np.linspace(1.18, 2.05, n_contours) # n
    contour_values_legend = []
    for i in range(len(contour_values)): contour_values_legend.append(str(contour_values[i]) + "ºC")

    contour_points = {}
    contour_points_arr = np.empty((n_contours, len(plot_times_zoomed)))
    # np.empty((len(contour_values), len(filtered_arr[:, 1])))
    nan_inds = {}
    real_inds = {}
    depths_arr = np.empty((len(contour_values), len(plot_times_zoomed)))
    cvi = 0
    nan_count = 0

    for cv in tqdm(contour_values):
        avg_zero_crossings = 0
        nan_inds[cvi] = []
        real_inds[cvi] = []

        for i in range(len(filtered_arr[:, 0])):
            xc = pyaC.zerocross1d(plot_dists_zoomed, filtered_arr[i, :] - cv)
            temp = filtered_arr[i, :] - cv
            # print(xc.flatten().tolist())
            if(len(xc) == 1):
                real_inds[cvi].append(i)
                contour_points_arr[cvi, i] = xc.flatten().tolist()[0]
            else:
                # print(xc)
                nan_inds[cvi].append(i)
                contour_points_arr[cvi, i] = float('NaN')
                
            avg_zero_crossings = avg_zero_crossings + len(xc)
        # print('Temp contour =', cv, 'avg of', avg_zero_crossings/len(filtered_arr[:, 0]), 'zero crossings')
        
        nan_inds[cvi] = np.array(nan_inds[cvi])
        real_inds[cvi] = np.array(real_inds[cvi])

        cvi = cvi + 1

    # Convert the contours into displacements 

    depths = np.empty(np.shape(contour_points_arr))
    displacements = np.empty(np.shape(contour_points_arr))
    median_sub_window = 144

    # filter = scipy.signal.butter(1, 0.5, 'hp', fs=5, output='sos', analog=False) # 5 represents 5 second sampling rate. Used to be 0.5 cutoff

    for i in tqdm(range(len(contour_points_arr[:, 0]))):
        real_inds = ~np.isnan(contour_points_arr[i, :])
        # avg_depth = np.interp(0, np.array(range(len(contour_points_arr[i, :])))[real_inds], contour_points_arr[i, real_inds])
        for j in range(len(contour_points_arr[i, :])):
            if(j < median_sub_window): avg_depth = np.nanmedian(contour_points_arr[i, 0:(j+median_sub_window)])
            elif(j > len(contour_points_arr[i, :] - median_sub_window)): avg_depth = np.nanmedian(contour_points_arr[i, (j-median_sub_window):])
            else: avg_depth = np.nanmedian(contour_points_arr[i, (j-median_sub_window):(j+median_sub_window)])
            displacements[i, j] = contour_points_arr[i, j] - avg_depth
            depths[i, j] = avg_depth
            # if(avg_depth == 0): print(contour_points_arr[i, (j-median_sub_window):(j+median_sub_window)])
        # disps_interp[i, :] = contour_points_arr[i, :] - contour_points_arr[i, 0]

    # Interpolate and gaussian filter each timestep
    disps_interp = np.empty( (n_contours, len(plot_times_zoomed)) )
    depths_interp = np.linspace(plot_dists_zoomed[0], plot_dists_zoomed[-1], len(contour_points_arr[:, 0]))

    # Interpolate and gaussian blur each vertical slice
    for i in range(len(displacements[0, :])):
        real_inds = ~np.isnan(displacements[:, i])
        disps_interp[:, i] = np.interp(depths_interp, depths[real_inds, i], displacements[real_inds, i] - np.nanmedian(displacements[real_inds, i]))
        disps_interp[:, i] = scipy.ndimage.gaussian_filter(disps_interp[:, i], 1)

    return disps_interp
