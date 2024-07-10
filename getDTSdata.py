def getDTSdata():
    # load packages
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import datetime
    import scipy

    # Read temperature data
    infile = '../data/DTS/temp_cal_valid_cable_rmnoise.csv'
    df_temp = pd.read_csv(infile, index_col=0)
    df_temp.columns = pd.to_datetime(df_temp.columns)
    df_temp.head()

    # Put temperatur data, optical distances and sampling times into numpy arrays
    temp_arr = df_temp.to_numpy().T # we want to have the time to be in the 0-axis to be consistent with DAS data
    times = pd.to_datetime(df_temp.columns)
    dists = df_temp.index.to_numpy()


    # plot selected DTS data over time
    # enter water at: 560m, exit water at 7565m
    start_dist = 2820 # east ridge at 2820m optical distance
    end_dist = 3150 # seafloor from ~3150m optical distance onward

    start_time = times[0] #datetime.datetime(2023,8,9)
    end_time = times[-1] #datetime.datetime(2023,8,30)

    t_idx_start = np.argmin(np.abs(times-start_time))
    t_idx_end = np.argmin(np.abs(times-end_time))
    d_idx_start = np.argmin(np.abs(dists-start_dist))
    d_idx_end = np.argmin(np.abs(dists-end_dist))

    plot_arr = temp_arr[t_idx_start:t_idx_end, d_idx_start:d_idx_end]
    plot_times = times[t_idx_start:t_idx_end]
    plot_dists = dists[d_idx_start:d_idx_end]

    return plot_times, plot_dists, plot_arr