# Contains functions to convert DTS data to displacement data as well as the function that gets coefficients for 
# displacement data using SINDy

import numpy as np
import scipy
from tqdm import tqdm
from PyAstronomy import pyaC
import pysindy as ps

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
    median_sub_window = 144 * 2 # 144

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

    return disps_interp, depths_interp

def get_coefs(x, t, disps_interp, sparsity, control_vars, control_vars_names):
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    u_sol = disps_interp
    v_sol = ps.SmoothedFiniteDifference(axis=1)._differentiate(u_sol, t=dt)
    
    # feature_value[:, 1] = ps.SmoothedFiniteDifference(axis=0)._differentiate(avg_temp, t=dt) 

    u = np.zeros((len(x), len(t), 2))
    u[:, :, 0] = u_sol
    u[:, :, 1] = v_sol
    u_dot = ps.SmoothedFiniteDifference(axis=1)._differentiate(u, t=dt)

    control_vars_filled = np.zeros((len(x), len(t), len(control_vars_names)))
    for j in range(len(x)): control_vars_filled[j, :, :] = control_vars

    library_functions = [
        lambda x: x,
        lambda x: x * x,
        lambda x, y: x * y
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + x,
        lambda x, y: x + y
    ]

    parameter_lib = ps.PolynomialLibrary(degree=1, include_bias=False)
    
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=x,
        include_bias=False,
        is_uniform=True,
        periodic=True
    )

    lib = ps.ParameterizedLibrary(
        feature_library=pde_lib,
        parameter_library=parameter_lib,
        num_features=2,
        num_parameters=len(control_vars_names)
    )

    optimizer = ps.STLSQ(
        threshold=sparsity, # 8e-2
        max_iter=10000,
        normalize_columns=True,
    )
    
    feat_names = ['u', '(du/dt)']
    feat_names.extend(control_vars_names)
    
    model = ps.SINDy(feature_library=lib, feature_names=feat_names, optimizer=optimizer)

    try: model.fit(u, x_dot=u_dot, u=control_vars_filled, quiet=True)
    except: return model.get_feature_names(), np.zeros(len(model.get_feature_names())), 0

    # Get the coefficients
    coefficients = model.coefficients()

    # Get the term names
    feature_names = model.get_feature_names()
    score = model.score(u,t=dt, u=control_vars_filled)

    return feature_names, coefficients[1, :], score