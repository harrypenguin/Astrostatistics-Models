import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def loaddata(simulation_names: list):
    """
    Load data from the simulation data files (in simulation_names) and interpolate the SFHs to 139 timesteps.
    Returns [sfhs, [logmass, arcsinh(presentsfr)]] for each galaxy in the simulation data.
    """
    work_dir = '/Users/pengzehao/Desktop/UofT/Astrostatistics/Iyer_etal_2020_SFH_data/'
    extn = '_sfhs_psds.mat'

    combined = []
    for sim_name in simulation_names:
        sim_data = sio.loadmat(work_dir + sim_name + extn)

        # Interpolating SFHs into 139 timesteps 
        x = np.linspace(0, 14, 139) # creating general array to act as universal timestep array, with 139 timesteps
        sfhs = np.zeros((len(sim_data['smallsfhs'].T), 139)) # creating a np array of all 0s of size (number of galaxies, 139)
        xp = np.linspace(0, 14, len(sim_data['smallsfhs'].T[0])) 
        for i in range(len(sim_data['smallsfhs'].T)):
            sfhs[i] = np.interp(x, xp, sim_data['smallsfhs'].T[i])
            
        presentsfr = sfhs[:, -1]  # Accessing the last time step for redshift 0 sfr
        logmass = np.array(sim_data['logmass'].ravel())
        if sim_name in ['Simba', 'Mufasa']:
            combined = combined + [[arr, [m, np.arcsinh(s)]] for arr, m, s in zip(sfhs, logmass, presentsfr) if m > 10]
        else:
            combined = combined + [[arr, [m, np.arcsinh(s)]] for arr, m, s in zip(sfhs, logmass, presentsfr) if m > 9]
        # Setting a cut off for mass, categorized by simulation
            
    return combined

def filter_zeroes(inputHistories, mass_presentsfr):
    """
    Filter out galaxies with zero mass and zero present SFR or zero SFH.
    Returns filtered inputHistories, filtered mass_presentsfr
    """
    zero_indices = np.array([i for i in range(len(inputHistories)) if np.trapz(inputHistories[i]) == 0])
    mask = np.ones(inputHistories.shape[0], dtype=bool)
    mask[zero_indices] = False

    return inputHistories[mask], mass_presentsfr[mask]
