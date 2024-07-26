import numpy as np
import matplotlib.pyplot as plt
import torch


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def pred_actual_data_vis(pred_mass, pred_sfr, stellar_mass, star_formation_rate):
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first set of histograms on the first subplot
    ax[0].hist(pred_sfr, bins=np.arange(0, 6, 0.05), alpha=0.5, label='Predicted', color='blue')
    ax[0].hist(star_formation_rate, bins=np.arange(0, 6, 0.05), alpha=0.5, label='Actual', color='orange')
    ax[0].set_xlabel('arcsinh Star Formation Rate ($M_\u2609$/Year)')
    ax[0].set_ylabel('Count')
    ax[0].set_ylim(0, 900)
    ax[0].set_title('arcsinh Redshift 0 Star Formation Rates (Predicted vs Actual)')
    ax[0].legend()
    
    # Plot the second set of histograms on the second subplot
    ax[1].hist(pred_mass, bins=np.arange(8.5, 12, 0.1), alpha=0.5, label='Predicted', color='blue')
    ax[1].hist(stellar_mass, bins=np.arange(8.5, 12, 0.1), alpha=0.5, label='Actual', color='orange')
    ax[1].set_xlabel('Log Stellar Mass ($M_\u2609$$)')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Log Stellar Mass (Predicted vs Actual)')
    ax[1].legend()
    
    # Display the plots
    plt.tight_layout()
    plt.show()
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First plot (2D Histogram of Stellar Mass vs Residuals)
    hist = ax[0].hist2d(stellar_mass, pred_mass - stellar_mass, bins=[np.linspace(5, 12, 100), np.linspace(-5, 5, 100)])
    ax[0].set_xlabel('Log Stellar Mass ($M_\u2609$)')
    ax[0].set_ylabel('Residual')
    ax[0].set_ylim(-5, 5)
    ax[0].set_xlim(8.5, 12)
    
    X, Y = np.meshgrid(np.linspace(5, 12, 100), np.linspace(-5, 5, 100))
    ax[0].contour(X[:-1, :-1] + np.diff(X[:2, 0]) / 2.,
                  Y[:-1, :-1] + np.diff(Y[0, :2]) / 2., hist[0].T, levels=6, colors='w', linewidths=0.5)
    
    hist = ax[1].hist2d(stellar_mass, pred_mass, bins=[np.linspace(5, 12, 500), np.linspace(5, 12, 500)])
    ax[1].set_xlabel('Actual Log Stellar Mass ($M_\u2609$)')
    ax[1].set_ylabel('Predicted Log Stellar Mass ($M_\u2609$)')
    ax[1].set_ylim(8.5, 12)
    ax[1].set_xlim(8.5, 12)
    ax[1].plot([8.5, 12], [8.5, 12], color='red', linestyle='--')
    cbar = plt.colorbar(hist[3], ax=ax[0])
    cbar.set_label('Counts')
    
    
    # Display the plots
    plt.tight_layout()
    plt.show()
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # First plot (2D Histogram of Stellar Mass vs Residuals)
    hist = ax[0].hist2d(star_formation_rate, pred_sfr - star_formation_rate, bins=[np.linspace(0.3, 1, 50), np.linspace(-5, 5, 100)])
    ax[0].set_xlabel('arcsinh SFR ($M_\u2609$/Year)')
    ax[0].set_xlim(0.3,1)
    ax[0].set_ylabel('Residual')
    
    X, Y = np.meshgrid(np.linspace(0.3, 1, 50), np.linspace(-5, 5, 100))
    ax[0].contour(X[:-1, :-1] + np.diff(X[:2, 0]) / 2.,
                  Y[:-1, :-1] + np.diff(Y[0, :2]) / 2., hist[0].T, levels=6, colors='w', linewidths=0.5)
    
    hist = ax[1].hist2d(star_formation_rate, pred_sfr, bins=[np.linspace(0.3, 5, 100), np.linspace(0.3, 5, 100)])
    ax[1].set_xlabel('Actual arcsinh SFR ($M_\u2609$/Year)')
    ax[1].set_ylabel('Predicted arcsinh SFR ($M_\u2609$/Year)')
    ax[1].set_xlim(0.3,5)
    ax[1].set_ylim(0.3,5)
    ax[1].plot([0.3, 5], [0.3, 5], color='red', linestyle='--')
    cbar = plt.colorbar(hist[3], ax=ax[0])
    cbar.set_label('Counts')
    
    
    # Display the plots
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # First plot (Actual Log Stellar Mass vs. arcsinh Star Formation Rate)
    ax[0].scatter(stellar_mass, star_formation_rate, s=0.1, alpha=1)
    ax[0].set_yscale('log')
    ax[0].set_xlim(8.5, 12)
    ax[0].set_ylim(1e-5, 10)
    ax[0].set_xlabel('Log Stellar Mass ($M_\u2609$)')
    ax[0].set_ylabel('arcsinh Star Formation Rate ($M_\u2609 Year^{-1}$)')
    ax[0].set_title('Actual Log Stellar Mass vs. arcsinh Star Formation Rate')
    
    # Second plot (Predicted Log Stellar Mass vs. arcsinh Star Formation Rate)
    ax[1].scatter(pred_mass, pred_sfr, s=0.1, alpha=1)
    ax[1].set_yscale('log')
    ax[1].set_xlim(8.5, 12)
    ax[1].set_ylim(1e-5, 10)
    ax[1].set_xlabel('Log Stellar Mass ($M_\u2609$)')
    ax[1].set_ylabel('arcsinh Star Formation Rate ($M_\u2609 Year^{-1}$)')
    ax[1].set_title('Predicted Log Stellar Mass vs. arcsinh Star Formation Rate')
    
    # Display the plots
    plt.tight_layout()
    plt.show()

def pred_actual_data_eval(pred_mass, pred_sfr, stellar_mass, star_formation_rate):
    r_squared_mass = r2_score(stellar_mass, pred_mass)
    r_squared_sfr = r2_score(star_formation_rate, pred_sfr)
    print(f'Score for log stellar mass: {r_squared_mass}\nScore for arcsinh SFR: {r_squared_sfr}')

    baseline_pred_mass = torch.tensor(np.full(len(stellar_mass), torch.mean(stellar_mass)))
    baseline_pred_sfr = torch.tensor(np.full(len(star_formation_rate), torch.mean(star_formation_rate)))
    print(f'\nRMSE for Stellar Mass: {torch.sqrt(torch.mean((pred_mass - stellar_mass)**2))} \nRMSE for SFR: {torch.sqrt(torch.mean((pred_sfr - star_formation_rate)**2))}') #Root mean squared error
    print(f'\nBaseline RMSE for Stellar Mass: {torch.sqrt(torch.mean((baseline_pred_mass - stellar_mass)**2))} (Predicting the mean) \nBaseline RMSE for SFR: {torch.sqrt(torch.mean((baseline_pred_sfr - star_formation_rate)**2))} (Predicting the mean)') #Root mean squared error

    print(f'\n1,5,10,50,90,95,99 Percentiles for mass residuals: {np.percentile(pred_mass - stellar_mass,[1,5,10,50,90,95,99])}') # Residual percentiles
    print(f'\n1,5,10,50,90,95,99 Percentiles for SFR residuals: {np.percentile(pred_sfr - star_formation_rate,[1,5,10,50,90,95,99])}') # Residual percentiles