import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_result

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from numpy.random import randn


def moving_average(data, window_size):
    '''
    '''
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def f_cv(x, dt, u):
    '''
    
    '''
    # Default parameters 
    egp = 1.07 # mg/dL/min
    egp = 0.18*2 # mg/dL/min
    gezi = 0.0035*0  # 1 / min
    vg = 199  # dL; volume of interstitial space for glucose
    tausen = 10  # min
    p2 = 0.0233  # 1/min
    ci = 909  # mL/min; insulin clearance rate
    tau1 = 71  # min
    tau2 = 70  # min
    tau_m = 50  # min

    # Unpack state and inputs 
    ndims = len(x)
    if ndims == 7:
        si = 4.63e-4  # mL/muU
        si = 0.0005  # mL/muU
        G, Ieff, Isc, Ip, Gisf, G1, G2 = x
    elif ndims == 8:
        G, Ieff, Isc, Ip, Gisf, G1, G2, si = x

    carb_rate, insulin_rate = u

    # Insulin dynamics (2 compartments)
    dIsc = -Isc/tau1 + insulin_rate/(tau1*ci)  # Eq 1
    dIp = -Ip/tau2 + Isc/tau2  # Eq 2
    dIeff = -p2*Ieff + p2*si*Ip  # Eq 3

    # Interstitial glucose dynamics
    dG1 = -G1/tau_m + carb_rate/vg
    dG2 = -G2/tau_m + G1/tau_m
    dG = -(gezi + Ieff)*G + egp + G2/tau_m  # Eq 4
    
    # Time-delayed iCGM measurement
    dGisf = -Gisf / tausen + G / tausen  # Eq 10
    
    # No expected change in Si from the model state
    dSi = 0

    dx = np.array([dG, dIeff, dIsc, dIp, dGisf, dG1, dG2])

    if ndims == 8:
        dx = np.hstack((dx, dSi))
    x1 = x + dx*dt

    return x1

def h_cv(x):
    '''
    '''
    return np.array([x[0]])

def main():
    result_path = '/Users/mconn/Library/CloudStorage/GoogleDrive-mark.connolly@tidepool.org/My Drive/projects/Sensitivity Analysis/processed_data/icgm_analysis_vp_35_65a71e48e64827646838032c8d29d3ac91ac36da94de018be6f668559ff4f9c2_tbg=70_sbg=150.tsv'
    (_, sim_results_df) = load_result(result_path, ext="tsv")
    bgs = np.array(sim_results_df['bg_sensor'])
    
    # result_path = '/Users/mconn/Library/CloudStorage/GoogleDrive-mark.connolly@tidepool.org/My Drive/projects/Sensitivity Analysis/processed_data/icgm_user_618.csv'
    # df = pd.read_csv(result_path)
    # glucose_df = df.iloc[0]
    # glucose_np = np.array(glucose_df)
    # bgs = glucose_np[2:].astype(float)
    
    # bgs[int(9200/5)] = 250
    sigma_x = 1
    y = [np.array([bg + randn()*sigma_x, ]) for bg in bgs]

    tt = np.array(range(len(y)))

    reported_bolus = np.array(sim_results_df['reported_bolus'])
    reported_bolus = np.where(np.isnan(reported_bolus), 0, reported_bolus)
    reported_bolus = reported_bolus * 1e6 / 5  # Scale units for a bolus delivered over 5 minutes

    temp_basal = np.array(sim_results_df['temp_basal'])
    temp_basal = np.where(np.isnan(temp_basal), 0, temp_basal)
    temp_basal = temp_basal * 1e6 / 60 * 5 # Scale units for a basal delivered over 5 minutes
        
    # Set up unscented Kalman filter
    bg_start = 70.0
    x =  np.array([bg_start, 1e-2*5, 50.0, 0.0, bg_start, 0.0, 0.0])
    x =  np.array([bg_start, 1e-2*5, 50.0, 0.0, bg_start, 0.0, 0.0, 1e-3])

    ndims = len(x)
    dt = 5

    points = MerweScaledSigmaPoints(n=ndims, alpha=.001, beta=2., kappa=0)
    ukf = UKF(dim_x=ndims, dim_z=1, fx=f_cv, hx=h_cv, dt=dt, points=points)

    ukf.x = x
    ukf.R = np.eye(1) * .3 # Uncertainty in the measurment (noise from iCGM)
    ukf.Q = np.eye(ndims) * 1e-7 # Uncertainty in the model
    
    if ndims == 8:
        ukf.Q = np.eye(ndims) * 1e-8 # Uncertainty in the model
        ukf.Q[7,7] = 0 #1e-12*0

    uxs = []

    # Set up moving average
    ma_steps = 3
    tt_ma = tt[ma_steps-1:]
    bg_ma = []

    # Set up exponential filter
    alpha = 0.6
    bg_exponential = []
    bg_exponential.append(y[0][0])

    for t in tt:
        
        bg = y[t]
        bolus = reported_bolus[t]
        basal = temp_basal[t]

        u = [0, bolus+basal]

        ukf.predict(u=u)
        ukf.update(bg)
        ukf.x = np.clip(ukf.x, 1e-6, None)  # Constrain after update
        uxs.append(ukf.x.copy())

        if t >= ma_steps-1:
            ma_window = y[(t-ma_steps+1):(t+1)]
            bg_ma.append(np.mean(ma_window))
        
        if t > 0:
            smooth_update = alpha * bg + (1 - alpha) * bg_exponential[t - 1]
            bg_exponential.append(smooth_update[0])

        if t == 800/5:
            tp = []
            bg_pred = []

            for i in range(72):
                tp.append(t+i)
                ukf.predict(u=u)
                prediction = ukf.x.copy()
                bg_pred.append(prediction)
            
            bg_pred = np.array(bg_pred)
            tp = np.array(tp)
            
            ukf.x = uxs[-1]

    uxs = np.array(uxs)
    bg_kf = uxs[:,0]
    
    _, ax = plt.subplots(3,1)
    
    ax[0].plot(tt*dt, y, linewidth=2, color='black', label='Sensor')
    ax[0].plot(tt*dt, bg_kf, linewidth=2, color='red', label='Kalman Filter')
    ax[0].plot(tt*dt, bg_exponential, linewidth=2, color='green', label='Exponential Average')
    ax[0].plot(tt_ma*dt, bg_ma, linewidth=2, color='blue', label='Moving Average')
    
    ax[0].plot(tp*dt, bg_pred[:,0], linestyle='--', linewidth=2, color='purple')
    ax[0].set_ylabel('Blood Glucose (mg/dL)')
    ax[0].set_xlabel('Minutes')

    ax[0].legend()
    
    if ndims == 8:
        si_kf = uxs[:,7]
        ax[1].plot(tt*dt, si_kf, linewidth=2, color='black', label='Si')
        ax[1].set_ylim([0, np.max(si_kf[20:])])
        ax[1].set_xlabel('Minutes')
        ax[1].set_ylabel('Si (mL/muU)')
        
    
    lags = np.arange(-(len(bg_ma)-1), len(y))
    ax[2].plot(lags, np.correlate(bg_ma, np.squeeze(y), mode='full'))
    # ax[2].plot(np.correlate(bg_kf, np.squeeze(y), mode='full'))
    plt.show()

if __name__ == "__main__":
    main()