import numpy as np

def homogeneous_poisson_timestamps(rate_hz, duration_ms):
    """
    Generate spike times for a homogeneous Poisson process.

    Parameters:
        rate_hz (float): desired firing rate in Hz.
        duration_ms (float): total time to generate over, in milliseconds.

    Returns:
        spike_times (list of float): spike timestamps in ms.
    """
    spike_times = []
    t = 100
    while t < duration_ms:
        isi = np.random.exponential(1000.0 / rate_hz)  # ISI in ms
        t += isi
        if t < duration_ms:
            spike_times.append(t)
    return spike_times


import numpy as np

# def inhomogeneous_poisson_timestamps(lambdas, win_length, dt=1.0):
#     """
#     Generate spike timestamps using an inhomogeneous Poisson process.
    
#     Parameters:
#         lambdas (list or np.ndarray): firing rates for each time window (in Hz or per timestep).
#         win_length (int): number of time steps per window.
#         dt (float): duration of each time step in ms (or your preferred time unit).
    
#     Returns:
#         spike_times (list of float): spike timestamps across all windows.
#     """
#     spike_times = []
#     for i, lambd in enumerate(lambdas):
#         num_points = np.random.poisson(lambd)
#         times_in_window = np.sort(np.random.uniform(0, win_length * dt, num_points))
#         times_in_window += i * win_length * dt
#         spike_times.extend(times_in_window.tolist())
#     return spike_times

def inhomogeneous_poisson_timestamps(lambdas, win_length, dt=1.0, start_time=0.0):
    """
    Generate spike timestamps using an inhomogeneous Poisson process.

    Parameters:
        lambdas (list or np.ndarray): firing rates for each time window (in Hz or per timestep).
        win_length (int): number of time steps per window.
        dt (float): duration of each time step in ms (or your preferred time unit).
        start_time (float): time offset (in ms) to start generating spikes.

    Returns:
        spike_times (list of float): spike timestamps across all windows.
    """
    spike_times = []
    for i, lambd in enumerate(lambdas):
        num_points = np.random.poisson(lambd)
        times_in_window = np.sort(np.random.uniform(0, win_length * dt, num_points))
        times_in_window += i * win_length * dt + start_time
        spike_times.extend(times_in_window.tolist())
    return spike_times

# Independent non-homogeneous poisson
# Generation: https://en.wikipedia.org/wiki/Poisson_point_process#Simulation
# 1. Generate num_points from pois(lambda * win_length)
# 2. Place points in each window uniformly randomly

def inhomogeneous_poisson_through_num_points(lambdas, win_length):
    t = np.zeros(len(lambdas) * win_length)
    for i, lambd in enumerate(lambdas):

        num_points = np.random.poisson(lambd)

        if num_points >= win_length:
            t[i * win_length : (i + 1) * win_length] = 1
            continue

        random_inds = np.random.choice(a = np.arange(win_length), size = num_points, replace = False)
        spikes = np.zeros(win_length)
        spikes[random_inds] = 1
        t[i * win_length : (i + 1) * win_length] = spikes

    return t

def inhomogeneous_poisson_through_interspike_intervals(lambdas, win_length):
    t = np.zeros(len(lambdas) * win_length)
    for i, lambd in enumerate(lambdas):
        spikes = np.zeros(win_length)

        j = int(np.random.exponential(lambd))
        while j < win_length:
            spikes[j] = 1
            j += int(np.random.exponential(lambd))

        t[i * win_length : (i + 1) * win_length] = spikes

    return t

def get_lambda_pois(num_spikes_per_second: int, win_length: int) -> float:
    '''
    Computes poisson's lambda given the desired number of spikes per second.
    '''
    return num_spikes_per_second * win_length / 1000

def get_num_spikes_per_interval(data, interval):
    num_spikes_per_interval = []
    for i in range(interval, len(data) + 1, interval):
        num_spikes_per_interval.append(np.sum(data[i - interval : i]))
    return num_spikes_per_interval