#Imports 
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import os
from obspy import read

# Seismic
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime as UTC
from obspy import Stream

#Plotting
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.gridspec as gridspec

def seismic_data(client1, 
                 network1, 
                 station1, 
                 location1, 
                 channel1, 
                 t_start, 
                 t_end, 
                 ref_stat: bool = False, 
                 user1=None, 
                 password1=None, 
                 client2=None, 
                 user2=None, 
                 password2=None, 
                 network2=None, 
                 station2=None, 
                 location2=None, 
                 channel2=None):

    """
    Gather seismic waveform data from FDSN clients,
    or from local files if they already exist, 
    from one or two specified groups of stations
    and stores it as a dictionary.

    Parameters:
    ----------
    client1 (str): 
        FDSN client URL for group 1. e.g 'IRIS', "AUSPASS".
    network1 (str): 
        Network code for group 1. e.g 'IU', 'AU'.
    station1 (str): 
        Station code for group 1. e.g 'CASY', 'CWA90'. 
    location1 (str): 
        Location code for group 1. e.g "00", "10". 
        Use '*' for all locations.
    channel1 (str): 
        Channel code for group 1. e.g 'BH?', 'NHE'. 
        Use '?' for all types of partly specified channel. 
        Use '*' for all channels.
    t_start (UTCDateTime): 
        Start time for data retrieval.
    t_end (UTCDateTime): 
        End time for data retrieval.
    ref_stat (bool): 
        True/False. If True, gather data from a second group of stations.
    "..."2:
        Identical parameters as those labeled '1' but now for group 2 if ref_stat is True.

    Returns:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    """
    
    # ----------------------------------------------------------------------
    # File check
    # ----------------------------------------------------------------------
    time = t_start.strftime("%Y_%m_%d")
    if ref_stat == True:
        title = f'{network1}_{network2}'
    else:
        title = f'{network1}'
    
    filename = f"station_data_{title}_{time}.mseed"

    if os.path.exists(filename):
        print(f"Reading existing file: {filename}")
        station_data = read(filename)

    else:
        print("File not found. Downloading data")

        # ----------------------------------------------------------------------
        # Group 1
        # ----------------------------------------------------------------------
        g1 = Client(base_url=client1, 
                    user=user1, 
                    password=password1) # FDSN client 

        station_data = g1.get_waveforms(network=network1, 
                                        station=station1, 
                                        location=location1, 
                                        channel=channel1, 
                                        starttime=t_start, 
                                        endtime=t_end) #Gather waveform data

        # ----------------------------------------------------------------------
        # Optional Group 2
        # ----------------------------------------------------------------------
        if ref_stat == True:
            g2 = Client(base_url=client2, 
                        user=user2, 
                        password=password2) # FDSN client 

            station_data2 = g2.get_waveforms(network=network2, 
                                            station=station2, 
                                            location=location2, 
                                            channel=channel2, 
                                            starttime=t_start, 
                                            endtime=t_end) #Gather waveform data

            station_data = station_data + station_data2
            station_data.merge() 

            
    # ----------------------------------------------------------------------
    # Write to a dictionary
    # ----------------------------------------------------------------------
    filename = f"station_data_{title}_{time}.mseed"
    station_data.write(filename, format="mseed")

    wave_dict = defaultdict(list) 

    for tr in station_data:
        station = (
        f"{tr.stats.network}."
        f"{tr.stats.station}."
        f"{tr.stats.location}"
        )
        wave_dict[station].append(tr) 

    return wave_dict    

def select_time(wave_dict, 
                t_start, 
                duration):
    
    """
    Select a specific time window from seismic waveform data stored in a dictionary without altering the original.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    t_start (UTCDateTime):
        Start time for the time window.
    duration (float):
        Duration of the time window in seconds.
    
    Returns:
    new_dict (dict):
        Dictionary containing selected seismic waveform data.
    """
    t_end = t_start + duration
    new_dict = defaultdict(list)
    for station_name in wave_dict:
        st = Stream(wave_dict[station_name]).copy()
        st.trim(starttime=t_start, endtime=t_end, pad=False)

        new_dict[station_name].extend(st.traces)
    return new_dict

def apply_filter(wave_dict, 
                 filter_type, 
                 freqmin=None, 
                 freqmax=None, 
                 freq=None, 
                 corners=4, 
                 zerophase=True):
    
    """
    Apply a filter to seismic waveform data stored in a dictionary without altering the original.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    filter_type (str):
        Type of filter to apply. 
        Options include 'bandpass', 'bandstop', 'lowpass', 'highpass'.
        'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently unsupported.
    freqmin (float):
        Minimum frequency for bandpass/bandstop filters.
    freqmax (float):
        Maximum frequency for bandpass/bandstop filters.
    freq (float):
        Cutoff frequency for lowpass/highpass filters.
    corners (int):
        Number of corners for the filter.
    zerophase (bool):
        True/False. If True, apply a zero-phase filter.

    Returns:
    filtered_dict (dict):
        Dictionary containing filtered seismic waveform data.
    """

    filtered_dict = {}

    for station_name, traces in wave_dict.items():
        st = Stream([tr.copy() for tr in traces])
        if filter_type in ('bandpass', 'bandstop'):
            st.filter(type=filter_type, 
                      freqmin=freqmin, 
                      freqmax=freqmax, 
                      corners=corners, 
                      zerophase=zerophase)
            
        elif filter_type in ('lowpass', 'highpass'): 
            st.filter(type=filter_type, 
                      freq=freq, 
                      corners=corners, 
                      zerophase=zerophase)
            
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}") #'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently not setup

        filtered_dict[station_name] = st.traces
        
    return filtered_dict

def apply_agc(wave_dict, 
              window_s):

    """
    Apply Automatic Gain Control (AGC) to seismic waveform data stored in a dictionary.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    window_s (float):
        AGC window length in seconds.
    
    Returns:
    agc_dict (dict):
        Dictionary containing AGC applied seismic waveform data.
    """
    agc_dict = defaultdict(list)
    for station in wave_dict:
        for tr in wave_dict[station]:
            agc_trace = tr.copy()
            sr = agc_trace.stats.sampling_rate  # Sampling rate
                
            N = int(window_s * sr) # Number of samples in the AGC window
            N = min(N, len(agc_trace.data)) # Ensure window does not exceed trace length
            if N < 3:
                N = 3  # Ensure at least 3 samples in the window
            data = agc_trace.data.astype(float) 
            rms = np.sqrt(np.convolve(data**2, np.ones(N)/N, mode='same')) # Compute RMS using AGC window
            rms[rms < 1e-10] = 1e-10 # Prevent division by zero
            agc_trace.data = data / rms 
            agc_dict[station].append(agc_trace)

    return agc_dict

def plot_streams(wave_dict):
    
    """
    Plot seismic waveform data stored in a dictionary.
    
    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    """
    
    for station_name in wave_dict:
        if not wave_dict[station_name]:    # Skip empty stations
            print(f"Skipping empty station: {station_name}")
            continue
        st = Stream(wave_dict[station_name])
        print(f"Plotting {station_name}")
        st.plot()

def polar_plot(wave_dict, 
               NS_channel, 
               EW_channel,
               peak_method='first', 
               col_n=4):

    """
    Create polar plots from seismic waveform data stored in a dictionary.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    EW_channel (list of str):
        Possible channel codes for East-West instrument component.
    peak_method (str):
        Method to identify peak: 'max' for maximum amplitude peak.
                                 'first', 'second', or 'third' for first, second, or third peak respectively.
    col_n (int):
        Number of columns in the plot grid.
    """
    
    station_list = list(wave_dict.keys())
    
    # Calculate number of rows needed for figure. Set number of columns in function call
    row_n = int(np.ceil(len(station_list) / col_n)) 

    # Prepare Figure
    fig = plt.figure(figsize=(9*col_n, 9*row_n))

    # Find appropriate NS and EW Channels from function input
    def find_channel(stream, options):
        for ch in options:
            tr = stream.select(channel=ch)
            if len(tr) > 0:
                return tr[0]
        return None
    
    for i, station in enumerate(station_list, start=1):
        print(f"Processing {station}...")
        st = Stream(wave_dict[station])
        st.sort(['channel'])
        trNS = find_channel(st, NS_channel) # Try to find NS channel from function input
        trEW = find_channel(st, EW_channel) # Try to find EW channel from function input

        if trNS is None or trEW is None:
            print(f"{station}: missing required channels (NS options: {NS_channel}, EW options: {EW_channel}), skipping.")
            continue

        # Match channel lengths
        n = min(len(trNS.data), len(trEW.data))
        NS = trNS.data[:n]
        EW = trEW.data[:n]

        # Apply peak normalization
        scale = np.max(np.sqrt(NS**2 + EW**2))
        NS_norm = NS / scale
        EW_norm = EW / scale

        # Gather polar coordinates
        theta = np.arctan2(NS_norm, EW_norm)
        theta_deg = np.degrees(theta) % 360 # Convert to degrees, adjust to orientation
     
        r = np.sqrt(NS_norm**2 + EW_norm**2)

        # Gather peaks
        peaks, _ = find_peaks(r)

        peak = None

        if len(peaks) == 0:
            print(f"{station}: no peaks found for given criteria.")

        else:
            peak_idx = {
                "max": np.argmax(r[peaks]),
                "first": 0,
                "second": 1,
                "third": 2}

            if peak_method not in peak_idx:
                print('Unknown peak_method: {peak_method}. ' \
                      'Options include: "first", "second", "third", "max".')
                      
            idx = peak_idx[peak_method]

            if idx < len(peaks):
                peak = peaks[idx]
                angle = theta_deg[peak]
                print(f"{station}: {peak_method} peak at angle = {angle:.2f}°")
            else:
                print(f"{station}: less than {idx+1} peaks found, no {peak_method} peak available.")
            

        # Create polar plot
        ax = fig.add_subplot(row_n, col_n, i, projection="polar")

        # Time array
        t = np.arange(len(r)) * trNS.stats.delta

        # Colour Mapping
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(t.min(), t.max())

        lc = LineCollection(
            segments,
            cmap=cm.get_cmap("turbo", 2048),      
            norm=norm,
            linewidth=2,
            alpha=0.6,
            label = 'Normalised Amplitude')

        lc.set_array(t)

        ax.add_collection(lc)

        # Colour Bar
        cbar = plt.colorbar(lc, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label("Time (s)")

        # For title
        time = trNS.stats.starttime.strftime("%Y-%m-%d %H:%M:%S")
        timespan = trNS.stats.endtime - trNS.stats.starttime

        # Plot the motion if first peak exists
        if peak is not None:
            ax.plot(theta[peak], 
                    r[peak], 
                    'ro', 
                    markersize=8, 
                    label="First Peak")

            # Legend
            ax.legend(
            loc="upper left",
            #bbox_to_anchor=(-1.3, 1.1),
            fontsize=10,
            frameon=True)

            # Add cardinal direction annotations
            rmax = ax.get_rmax()
            cardinals = {
                "E": (0, rmax * 1.05),
                "N": (np.pi / 2, rmax * 1.05),
                "W": (np.pi, rmax * 1.05),
                "S": (3 * np.pi / 2, rmax * 1.05)}

            offset = 0.115 * rmax  # Offset for cardinal labels

            for label, (angle, radius) in cardinals.items():
                ax.text(
                    angle,
                    radius + offset,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False)
            
              
            ax.set_title(f"Horizontal Particle Motion Plot for {station} at {time} for {timespan} seconds", y=1.15)    

    plt.tight_layout()
    plt.savefig('seismic_polar_plots.png', dpi=300)
    plt.show()

def waveform_and_polar(wave_dict,  
                       NS_channel, 
                       EW_channel,
                       peak_method='first',
                       save_png=True):

    """
    Create waveform plots and polar plots from seismic waveform data stored in a dictionary.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    EW_channel (list of str):
        Possible channel codes for East-West instrument component.
    peak_method (str):
        Method to identify peak: 'max' for maximum amplitude peak.
                                 'first', 'second', or 'third' for first, second, or third peak respectively.
    save_png (bool):
        True/False. If True, save each plot as a PNG file.
    """
    
    station_list = list(wave_dict.keys())

    # Find appropriate NS and EW Channels from function input
    def find_channel(stream, options):
        for ch in options:
            tr = stream.select(channel=ch)
            if len(tr) > 0:
                return tr[0]
        return None
    
    for station in station_list:
        print(f"Processing {station}...")
        st = Stream(wave_dict[station])
        st.sort(['channel'])
        trNS = find_channel(st, NS_channel) # Try to find NS channel from function input
        trEW = find_channel(st, EW_channel) # Try to find EW channel from function input

        if trNS is None or trEW is None:
            print(f"{station}: missing required channels (NS options: {NS_channel}, EW options: {EW_channel}), skipping.")
            continue

        # Match channel lengths
        n = min(len(trNS.data), len(trEW.data))
        NS = trNS.data[:n]
        EW = trEW.data[:n]

        # Apply peak normalization
        scale = np.max(np.sqrt(NS**2 + EW**2))
        NS_norm = NS / scale
        EW_norm = EW / scale

        # Gather polar coordinates
        theta = np.arctan2(NS_norm, EW_norm)
        theta_deg = np.degrees(theta) % 360 # Convert to degrees, adjust to orientation
     
        r = np.sqrt(NS_norm**2 + EW_norm**2)

        # Gather peaks
        peaks, _ = find_peaks(r)

        peak = None

        if len(peaks) == 0:
            print(f"{station}: no peaks found for given criteria.")

        else:
            peak_idx = {
                "max": np.argmax(r[peaks]),
                "first": 0,
                "second": 1,
                "third": 2}

            if peak_method not in peak_idx:
                print('Unknown peak_method: {peak_method}. ' \
                      'Options include: "first", "second", "third", "max".')
                      
            idx = peak_idx[peak_method]

            if idx < len(peaks):
                peak = peaks[idx]
                angle = theta_deg[peak]
                print(f"{station}: {peak_method} peak at angle = {angle:.2f}°")
            else:
                print(f"{station}: less than {idx+1} peaks found, no {peak_method} peak available.")

        # Create plot
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 0.8], hspace=0.3)

        ax = fig.add_subplot(gs[0], projection="polar")   # particle motion
        ax_w = fig.add_subplot(gs[1])                     # waveform


        # Time array
        t = np.arange(len(r)) * trNS.stats.delta

        
        # Colour Mapping
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(t.min(), t.max())

        lc = LineCollection(
            segments,
            cmap=cm.get_cmap("turbo", 2048),      
            norm=norm,
            linewidth=2,
            alpha=0.6,
            label = 'Normalised Amplitude')

        lc.set_array(t)

        ax.add_collection(lc)
        ax.set_rlim(0, 1.05)


        # Colour Bar
        cbar = plt.colorbar(lc, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label("Time (s)")

        # For title
        time = trNS.stats.starttime.strftime("%Y-%m-%d %H:%M:%S")
        timespan = trNS.stats.endtime - trNS.stats.starttime

        # Plot the motion if first peak exists
        if peak is not None:
            ax.plot(theta[peak], 
                    r[peak], 
                    'ro', 
                    markersize=8, 
                    label="Peak")

            # Legend
            ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.3, 1.1),
            fontsize=10,
            frameon=True)

            # Add cardinal direction annotations
            rmax = ax.get_rmax()
            cardinals = {
                "E": (0, rmax * 1.05),
                "N": (np.pi / 2, rmax * 1.05),
                "W": (np.pi, rmax * 1.05),
                "S": (3 * np.pi / 2, rmax * 1.05)}

            offset = 0.17 * rmax  # Offset for cardinal labels

            for label, (angle, radius) in cardinals.items():
                ax.text(
                    angle,
                    radius + offset,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False)
            
            # Waveform Plot
            ax_w.plot(t, NS_norm, label=trNS.stats.channel, linewidth=1)
            ax_w.plot(t, EW_norm, label=trEW.stats.channel, linewidth=1)

            ax_w.set_xlabel("Time (s)")
            ax_w.set_ylabel("Normalised Amplitude")
            ax_w.legend()
            ax_w.grid(True, alpha=0.3)  
            fig.suptitle(
            f"Waveform Plot for {station} at {time} for {timespan} seconds", y=.3)

            ax.set_title(f"Horizontal Particle Motion Plot for {station} at {time} for {timespan} seconds", y=1.15)  

            plt.tight_layout()
            
            if save_png == True:
                plt.savefig(f'seismic_wfandpolar_{station}.png', dpi=300)

            plt.show()
            plt.close(fig)

    plt.show()

def tabulate_peaks(wave_dict, 
                   NS_channel, 
                   EW_channel,
                   location='default_title'):
    
    """
    Tabulate angles of first peaks from seismic waveform data stored in a dictionary.
    
    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    EW_channel (list of str):
        Possible channel codes for East-West instrument component.  
    location (str):
        Title/location for the output table and CSV file.

    Returns:
    df (DataFrame):
        DataFrame containing peak angles for each station.
    """
    
    angle_results = []   # table storage
    station_list = list(wave_dict.keys())

    # Find appropriate NS and EW Channels from function input
    def find_channel(stream, options):
        for ch in options:
            tr = stream.select(channel=ch)
            if len(tr) > 0:
                return tr[0]
        return None
        
    for i, station in enumerate(station_list, start=1):
        print(f"Processing {station}...")
        st = Stream(wave_dict[station])
        st.sort(['channel'])
        trNS = find_channel(st, NS_channel) # Try to find NS channel from function input
        trEW = find_channel(st, EW_channel) # Try to find EW channel from function input

        if trNS is None or trEW is None:
            print(f"{station}: missing required channels (NS options: {NS_channel}, EW options: {EW_channel}), skipping.")
            continue

        # Match channel lengths
        n = min(len(trNS.data), len(trEW.data))
        NS = trNS.data[:n]
        EW = trEW.data[:n]

        # Apply peak normalization
        scale = np.max(np.sqrt(NS**2 + EW**2))
        NS_norm = NS / scale
        EW_norm = EW / scale

        # Gather polar coordinates
        theta = np.arctan2(NS_norm, EW_norm)
        theta_deg = np.degrees(theta) % 360 # Convert to degrees, adjust to orientation
        r = np.sqrt(NS_norm**2 + EW_norm**2)

        # Gather peaks
        peaks, _ = find_peaks(r)

        # Default values
        peak_angles = {
            "Max Peak (°)": np.nan,
            "First Peak (°)": np.nan,
            "Second Peak (°)": np.nan,
            "Third Peak (°)": np.nan,}
        
        if len(peaks) > 0:
            peak_angles["Max Peak (°)"] = theta_deg[peaks[np.argmax(r[peaks])]]

            for i, key in enumerate(["First Peak (°)", "Second Peak (°)", "Third Peak (°)"]):
                if i < len(peaks):
                    peak_angles[key] = theta_deg[peaks[i]]

        # Store results in table
        angle_results.append({"Station": station,**peak_angles})

    # Tabulate
    time = trNS.stats.starttime
    df = pd.DataFrame(angle_results)
    df.to_csv(f'seismic_directions_{location}.csv', index=False)
    print(f"Peaks for {location} Earthquake @ {time} (UTC):")
    print(df.to_string(index=False))

    return df

def polar_correction(wave_dict, 
                     NS_channel, 
                     EW_channel,
                     peak_method='first', 
                     ref_angle=0,
                     col_n=4,):

    """
    Create polar plots from seismic waveform data stored in a dictionary.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    EW_channel (list of str):
        Possible channel codes for East-West instrument component.
    peak_method (str):
        Method to identify peak: 'max' for maximum amplitude peak.
                                 'first', 'second', or 'third' for first, second, or third peak respectively.
    ref_angle (float):
        Angle in radians to rotate the polar plot.
    col_n (int):
        Number of columns in the plot grid.
    """
    
    station_list = list(wave_dict.keys())
    
    # Calculate number of rows needed for figure. Set number of columns in function call
    row_n = int(np.ceil(len(station_list) / col_n)) 

    # Prepare Figure
    fig = plt.figure(figsize=(6*col_n, 6*row_n))

    # Find appropriate NS and EW Channels from function input
    def find_channel(stream, options):
        for ch in options:
            tr = stream.select(channel=ch)
            if len(tr) > 0:
                return tr[0]
        return None
    
    for i, station in enumerate(station_list, start=1):
        print(f"Processing {station}...")
        st = Stream(wave_dict[station])
        st.sort(['channel'])
        trNS = find_channel(st, NS_channel) # Try to find NS channel from function input
        trEW = find_channel(st, EW_channel) # Try to find EW channel from function input

        if trNS is None or trEW is None:
            print(f"{station}: missing required channels (NS options: {NS_channel}, EW options: {EW_channel}), skipping.")
            continue

        # Match channel lengths
        n = min(len(trNS.data), len(trEW.data))
        NS = trNS.data[:n]
        EW = trEW.data[:n]

        # Apply peak normalization
        scale = np.max(np.sqrt(NS**2 + EW**2))
        NS_norm = NS / scale
        EW_norm = EW / scale

        # Gather polar coordinates
        theta = np.arctan2(NS_norm, EW_norm)
        theta_deg = np.degrees(theta) % 360 # Convert to degrees, adjust to orientation
        r = np.sqrt(NS_norm**2 + EW_norm**2)

        # Gather peaks
        peaks, _ = find_peaks(r)

        peak = None

        if len(peaks) == 0:
            print(f"{station}: no peaks found for given criteria.")

        else:
            peak_idx = {
                "max": np.argmax(r[peaks]),
                "first": 0,
                "second": 1,
                "third": 2}

            if peak_method not in peak_idx:
                print('Unknown peak_method: {peak_method}. ' \
                      'Options include: "first", "second", "third", "max".')
                      
            idx = peak_idx[peak_method]

            if idx < len(peaks):
                peak = peaks[idx]
                angle = theta_deg[peak]
                print(f"{station}: {peak_method} peak at angle = {angle:.2f}°")
            else:
                print(f"{station}: less than {idx+1} peaks found, no {peak_method} peak available.")

        # Angle correction
        if peak is not None:
            correction_angle = theta[peak] - ref_angle
        else:
            correction_angle = 0.0

        # Create polar plot-
        ax = fig.add_subplot(row_n, col_n, i, projection="polar")
        ax.plot(theta, 
                r, 
                alpha=0.30, 
                color = 'blue',
                label="Normalised Amplitude")
        ax.plot(theta - correction_angle, 
                r, 
                alpha=0.65, 
                color = 'orange',
                label="Normalised Amplitude Corrected")
        
        # For title
        time = trNS.stats.starttime.strftime("%Y-%m-%d %H:%M:%S")
        timespan = trNS.stats.endtime - trNS.stats.starttime

        # Plot the motion if peak exists
        if peak is not None:
            ax.plot(theta[peak], 
                    r[peak], 
                    'ro', 
                    color = 'pink',
                    alpha = 0.8,
                    markersize=8, 
                    label="Peak")
            ax.plot(theta[peak] - correction_angle, 
                    r[peak], 
                    'ro', 
                    color = 'red',
                    markersize=8, 
                    label=f"{peak_method} Peak Corrected")

            # Legend
            ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            fontsize=8,
            frameon=True)

            # Add cardinal direction annotations
            rmax = ax.get_rmax()
            cardinals = {
                "E": (0, 1.05 * rmax * 1.05),
                "N": (np.pi / 2, rmax * 1.05),
                "W": (np.pi, 1.05 * rmax * 1.05),
                "S": (3 * np.pi / 2, rmax * 1.05)}

            offset = 0.115 * rmax  # Offset for cardinal labels

            for label, (angle, radius) in cardinals.items():
                ax.text(
                    angle,
                    radius + offset,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False)
            
              
            ax.set_title(f"Horizontal Particle Motion Plot \n for {station} at {time} for {timespan} seconds", y=1.15)    

    plt.tight_layout()
    plt.savefig('polar_correction_plots.png', dpi=300)
    plt.show()
