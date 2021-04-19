import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from constants import *
from initialize import init
from electronic import electronic
from dynamic_switching import switching
from tmr import tmr
# import multiprocessing as mp #parallelization to speed up computation
import time as t


def manual_main_1():
    """
    From manual:

    Firstly apply constant voltage of 1V on V1 and
    constant voltage of 0.1V on V3 (from 0 to 2ns). Set ESTT to be 0(STT effect
    does not exist).Set ESOT to be 1(SOT effect exists).Set Ratio of field-like
    (FL) torque to damping-like (DL) torque in SOT effect to be 0.83 .
    As in the matlab source code

    """
    # Time range:
    sim_startup = 1  # 1st stage
    sim_mid = 2000  # 2nd stage
    sim_end = 5000
    # t_step = 1e-12 s  = 1 ps
    time = np.arange(sim_startup-1, (sim_end+1))  # time step

    # Input parameters, init to all 0s, save per t_step:
    # matlab index inclusive
    M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
    Theta = np.zeros(sim_end+1)
    Phi = np.zeros(sim_end+1)
    R = np.zeros(sim_end+1)
    V = np.zeros(sim_end+1)
    PAP = 1

    R_MTJ, theta, mz, phi = init(PAP)
    R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

    V1_1, V2_1, V3_1 = 1, 0, 0.1  # 1st stage
    V1_2, V2_2, V3_2 = -1, 0, 0  # 2nd stage
    V1_3, V2_3, V3_3 = 0, 0, 0  # 3rd stage

    # 1st stage
    for i in range(sim_startup-1, sim_mid):  # 2000 times iteration from 0th to 1999th
        V1, V2, V3 = V1_1, V2_1, V3_1
        ESTT, ESOT = 0, 1  # as in the description
        R_MTJ = R[i]
        I_SOT, V_MTJ = electronic(V1, V2, V3, R_MTJ)
        mz, phi_temp, theta_temp = switching(
            V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=1, R_SOT_FL_DL=0.83)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ, mz)

        V[i] = V_MTJ
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    # 3rd stage
    for i in range(sim_mid, sim_end):
        V1, V2, V3 = V1_3, V2_3, V3_3
        ESOT, ESTT = 1, 1  # as in description
        R_MTJ = R[i]
        I_SOT, V_MTJ = electronic(V1, V2, V3, R_MTJ)
        mz, phi_temp, theta_temp = switching(
            V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=1, R_SOT_FL_DL=0.83)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ, mz)

        V[i] = V_MTJ
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    for i in range(sim_startup-1, sim_end):
        V_MTJ = 0
        mz = M_z[i]
        R_MTJ = tmr(V_MTJ, mz)
        R[i+1] = R_MTJ

    fig, axs = plt.subplots(2)
    tick_spacing = 5e-10
    axs[0].plot(time*t_step, M_z)  # mz vs time
    axs[0].set(xlabel='time(s)', ylabel='mz')
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axs[1].plot(time*t_step, R)  # R_MTJ vs time
    axs[1].set(xlabel='time(s)', ylabel='R_MTJ')
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig.tight_layout()
    axs[0].grid()
    axs[1].grid()
    plt.gcf().canvas.set_window_title('manual_main_1')
    # plt.show()
    plt.savefig('result/manual_main_1.png')
    plt.close()


def manual_main_2():
    """
    From manual:

    Firstly apply constant voltage of 0 V on VMTJ and
    constant current of -95e-6 on ISOT (from 0 to 2ns). Set ESTT to be 0(STT
    effect does not exist).Set ESOT to be 1(SOT effect exists).Set Ratio of field
    -like (FL) torque to damping-like (DL) torque in SOT effect to be 0.83 

    Warning:
    - Change R_SOT_FL_DL to 0.83 in constants.py
    - Change VNV = 1 in dynamic_switching.py
    """
    V_MTJ_1, V_MTJ_2, V_MTJ_3 = 0, 0, 0
    I_SOT_1, I_SOT_2, I_SOT_3 = -95e-6, 0, 0

    # Time range:
    sim_startup = 1  # 1st stage
    sim_mid1 = 2000  # 2nd stage
    sim_end = 5000
    # t_step = 1e-12 s  = 1 ps
    time = np.arange(sim_startup-1, (sim_end+1))  # time step

    # Input parameters, init to all 0s, save per t_step:
    # matlab index inclusive
    M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
    Theta = np.zeros(sim_end+1)
    Phi = np.zeros(sim_end+1)
    R = np.zeros(sim_end+1)
    V = np.zeros(sim_end+1)
    PAP = 1

    R_MTJ, theta, mz, phi = init(PAP)
    R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

    # 1st stage
    for i in range(sim_startup-1, sim_mid1):
        V_MTJ, I_SOT = V_MTJ_1, I_SOT_1
        ESTT, ESOT = 0, 1
        R_MTJ = R[i]

        mz, phi_temp, theta_temp = switching(
            V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=1, R_SOT_FL_DL=0.83)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ, mz)

        V[i] = V_MTJ
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    # 2nd stage
    for i in range(sim_mid1, sim_end):
        V_MTJ, I_SOT = V_MTJ_2, I_SOT_2
        ESTT, ESOT = 0, 1
        R_MTJ = R[i]

        mz, phi_temp, theta_temp = switching(
            V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=1, R_SOT_FL_DL=0.83)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ, mz)

        V[i] = V_MTJ
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    tick_spacing = 5e-10
    plt.plot(time*t_step, M_z)  # mz vs time
    plt.xlabel('time(s)')
    plt.ylabel('mz')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.grid()
    plt.gcf().canvas.set_window_title('manual_main_2')
    # plt.show()
    plt.savefig('result/manual_main_2.png')
    plt.close()


def paper_main_switch_without_VCMA(NON=0):
    """
    Same idea as main_2()
    To reproduce result in paper (see III A Figure 3a)
    But with I_SOT from -85uA, -90uA, -95uA, and -100uA

    Warning:
    - Change R_SOT_FL_DL to 0 in constants.py
    - Change VNV = 0 in dynamic_switching.py
    """
    I_SOT_matrix = [-85e-6, -90e-6, -95e-6, -100e-6]

    for I_SOT_i in I_SOT_matrix:
        V_MTJ_1, V_MTJ_2, V_MTJ_3 = 0, 0, 0
        I_SOT_1, I_SOT_2, I_SOT_3 = I_SOT_i, 0, 0

        # Time range:
        sim_startup = 1  # 1st stage
        sim_mid1 = 2000  # 2nd stage
        sim_end = 5000
        # t_step = 1e-12 s  = 1 ps
        time = np.arange(sim_startup-1, (sim_end+1))  # time step

        # Input parameters, init to all 0s, save per t_step:
        # matlab index inclusive
        M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
        Theta = np.zeros(sim_end+1)
        Phi = np.zeros(sim_end+1)
        R = np.zeros(sim_end+1)
        V = np.zeros(sim_end+1)
        PAP = 1

        R_MTJ, theta, mz, phi = init(PAP)
        R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

        # 1st stage
        for i in range(sim_startup-1, sim_mid1):
            V_MTJ, I_SOT = V_MTJ_1, I_SOT_1
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=0, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ, mz)

            V[i] = V_MTJ
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        # 2nd stage
        for i in range(sim_mid1, sim_end):
            V_MTJ, I_SOT = V_MTJ_2, I_SOT_2
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=0, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ, mz)

            V[i] = V_MTJ
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        tick_spacing = 5e-10
        # mz vs time
        plt.plot(time*t_step, M_z, label='I_SOT = {} uA'.format(I_SOT_i))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    plt.xlabel('time(s)')
    plt.ylabel('mz')
    plt.tight_layout()
    plt.grid()
    plt.legend(loc='upper left', title='H_EX = -50 Oe')
    if(NON == 0):
        plt.gcf().canvas.set_window_title('paper_main_switch_without_VCMA')
        plt.savefig('result/paper_main_switch_without_VCMA.png')
        plt.close()
    elif(NON == 1):
        plt.gcf().canvas.set_window_title('paper_main_switch_without_VCMA_with_noise')
        plt.savefig('result/paper_main_switch_without_VCMA_with_noise.png')
        plt.close()
    # plt.show()


def paper_main_switch_without_VCMA_with_noise():
    """
    same as paper_main_switch_without_VCMA
    see section III A Figure 3b

    Can give different results due to random Gaussian thermal noise
    """
    paper_main_switch_without_VCMA(NON=1)  # thermal noise considered


def switch_error_helper_function(I_SOT, NON):
    """
    return average error rate
    for SER (next function)
    """
    V_MTJ_1, V_MTJ_2, V_MTJ_3 = 0, 0, 0
    I_SOT_1, I_SOT_2, I_SOT_3 = I_SOT, 0, 0
    error_sum = 0

    start_time = t.time()

    for k in range(1000):
        # Monte-Carlo 1000 times simulation

        # Time range:
        sim_startup = 1  # 1st stage
        sim_mid1 = 2000  # 2nd stage
        sim_end = 5000
        # t_step = 1e-12 s  = 1 ps
        time = np.arange(sim_startup-1, (sim_end+1))  # time step

        # Input parameters, init to all 0s, save per t_step:
        # matlab index inclusive
        # sim_end+1 elements from 0th to sim_end-th
        M_z = np.zeros(sim_end+1)
        Theta = np.zeros(sim_end+1)
        Phi = np.zeros(sim_end+1)
        R = np.zeros(sim_end+1)
        V = np.zeros(sim_end+1)
        PAP = 1

        R_MTJ, theta, mz, phi = init(PAP)
        R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

        # 1st stage
        for i in range(sim_startup-1, sim_mid1):
            V_MTJ, I_SOT = V_MTJ_1, I_SOT_1
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=0, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ, mz)

            V[i] = V_MTJ
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        # 2nd stage
        for i in range(sim_mid1, sim_end):
            V_MTJ, I_SOT = V_MTJ_2, I_SOT_2
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=0, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ, mz)

            V[i] = V_MTJ
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        if(M_z[sim_end] - (-1.0) < 1e-1):  # switching failure
            error_sum += 1

    print('I_SOT = {}A, Elapsed = {}s'.format(I_SOT, t.time()-start_time))

    average = error_sum/1000
    return average


def paper_switch_error_withoutVCMA_withNoise(NON=1):
    """
    uses 1000 iterations for every I_SOT value
    To reproduce section III A Figure 3c

    Warning: might take around 45 minutes
    """
    I_SOT_matrix = [-100e-6, -98e-6, -96e-6, -94e-6, -92e-6, -90e-6]

    # start_time = t.time()
    # pool = mp.Pool(mp.cpu_count())
    # error_rate_matrix = [pool.starmap(switch_error_helper_function, [(I_SOT, NON)]) for I_SOT in I_SOT_matrix]
    # print('Elapsed: {}'.format(t.time()-start_time))

    error_rate_matrix = []
    for I_SOT_i in I_SOT_matrix:
        start_time = t.time()
        average = switch_error_helper_function(I_SOT_i, NON)
        print("{} seconds".format(t.time() - start_time))
        error_rate_matrix.append(average)

    # start_time = t.time()
    # error_rate_matrix = []
    # ufunc_helper = np.frompyfunc(switch_error_helper_function, 2, 1)
    # error_rate_matrix = ufunc_helper(I_SOT_matrix, [NON, NON, NON, NON, NON, NON])
    # print('Elapsed: {}'.format(t.time()-start_time))

    plt.plot(I_SOT_matrix, error_rate_matrix, marker = '0')  # error vs I_SOT
    plt.xlabel('I_SOT')
    plt.ylabel('SER')
    plt.tight_layout()
    plt.grid()
    plt.gcf().canvas.set_window_title('switch_error_withoutVCMA_withNoise')
    # plt.show()
    plt.savefig('result/switch_error_withoutVCMA_withNoise.png')
    plt.close()


def paper_main_switch_with_VCMA_different_ISOT(NON=0, VNV=1):
    """
    To reproduce result in paper section III C Figure 5a
    VCMA-Enabled -> VNV = 1

    H_EX = -50 Oe
    V_MTJ = 1.2 V
    I_SOT from -90e-6, -30e-6,  -18e-6, -16e-6
    From 0 to 25 ns
    """
    I_SOT_matrix = [-90e-6, -30e-6,  -18e-6, -16e-6]
    V_MTJ = 1.2

    for I_SOT_i in I_SOT_matrix:

        # Time range:
        sim_startup = 1  # 1st stage
        sim_end = 25000
        time = np.arange(sim_startup-1, (sim_end+1))  # time step

        # Input parameters, init to all 0s, save per t_step:
        # matlab index inclusive
        M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
        Theta = np.zeros(sim_end+1)
        Phi = np.zeros(sim_end+1)
        R = np.zeros(sim_end+1)
        V = np.zeros(sim_end+1)
        PAP = 1

        R_MTJ, theta, mz, phi = init(PAP)
        R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

        # 1st stage
        for i in range(sim_startup-1, sim_end):
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ, I_SOT_i, R_MTJ, theta, phi, ESTT, ESOT, VNV=VNV, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ, mz)

            V[i] = V_MTJ
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        tick_spacing = 5e-9
        # mz vs time
        plt.plot(time*t_step, M_z, label='I_SOT = {} uA'.format(I_SOT_i))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    plt.xlabel('time(s)')
    plt.ylabel('mz')
    plt.tight_layout()
    plt.grid()
    plt.legend(loc='lower right', title='H_EX = -50 Oe')
    plt.gcf().canvas.set_window_title('paper_main_switch_with_VCMA_different_ISOT')
    plt.savefig('result/paper_main_switch_with_VCMA_different_ISOT.png')
    plt.close()
    # plt.show()


def paper_main_switch_with_VCMA_different_V_MTJ(NON=0, VNV=1):
    """
    To reproduce result in paper section III C Figure 5c
    VCMA-Enabled -> VNV = 1

    H_EX = -50 Oe = H_DL_SOT
    -> I_SOT =  (2*e*u0*Ms*tf*A2*(-50*1000/(4*pi)))/(h_bar*theta_SH)
    V_MTJ from 1.3189, 1.3191, 1.333, 1.3489, 1.4937
    From 0 to 25 ns
    """
    I_SOT = (2*e*u0*Ms*tf*A2*(-50*1000/(4*pi)))/(h_bar*theta_SH)  # -6.2619uA
    V_MTJ_matrix = [1.3189, 1.3191, 1.333, 1.3489, 1.4937]

    for V_MTJ_i in V_MTJ_matrix:

        # Time range:
        sim_startup = 1  # 1st stage
        sim_end = 25000
        time = np.arange(sim_startup-1, (sim_end+1))  # time step

        # Input parameters, init to all 0s, save per t_step:
        # matlab index inclusive
        M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
        Theta = np.zeros(sim_end+1)
        Phi = np.zeros(sim_end+1)
        R = np.zeros(sim_end+1)
        V = np.zeros(sim_end+1)
        PAP = 1

        R_MTJ, theta, mz, phi = init(PAP)
        R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

        # 1st stage
        for i in range(sim_startup-1, sim_end):
            ESTT, ESOT = 0, 1
            R_MTJ = R[i]

            mz, phi_temp, theta_temp = switching(
                V_MTJ_i, I_SOT, R_MTJ, theta, phi, ESTT, ESOT, VNV=VNV, NON=NON, R_SOT_FL_DL=0)
            phi, theta = phi_temp, theta_temp
            R_MTJ = tmr(V_MTJ_i, mz)

            V[i] = V_MTJ_i
            # for next iteration
            M_z[i+1] = mz
            Theta[i+1] = theta
            Phi[i+1] = phi
            R[i+1] = R_MTJ

        tick_spacing = 5e-9
        # mz vs time
        plt.plot(time*t_step, M_z, label='V_MTJ = {} V'.format(V_MTJ_i))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    plt.xlabel('time(s)')
    plt.ylabel('mz')
    plt.tight_layout()
    plt.grid()
    plt.legend(loc='center right', title='H_EX = -50 Oe, \nI_SOT = -6.2619uA')
    plt.gcf().canvas.set_window_title('paper_main_switch_with_VCMA_different_V_MTJ')
    plt.savefig('result/paper_main_switch_with_VCMA_different_V_MTJ.png')
    plt.close()
    # plt.show()


def paper_proposed_main_switch_helper(t1, t2, V_MTJ_1, V_MTJ_2, I_SOT_1, I_SOT_2, NON=0, VNV=1):
    """
    Helper function for proposed_main_switch()
    return time*t_step, M_z
    """
    # Time range:
    sim_startup = 1  # 1st stage
    sim_mid1 = int(t1/t_step)  # 2nd stage
    sim_mid2 = int((t1+t2)/t_step)
    sim_end = 25000
    # t_step = 1e-12 s  = 1 ps
    time = np.arange(sim_startup-1, (sim_end+1))  # time step

    # Input parameters, init to all 0s, save per t_step:
    # matlab index inclusive
    M_z = np.zeros(sim_end+1)  # sim_end+1 elements from 0th to sim_end-th
    Theta = np.zeros(sim_end+1)
    Phi = np.zeros(sim_end+1)
    R = np.zeros(sim_end+1)
    V = np.zeros(sim_end+1)
    PAP = 1
    ESTT, ESOT = 0, 1

    R_MTJ, theta, mz, phi = init(PAP)
    R[0], Theta[0], M_z[0], Phi[0] = R_MTJ, theta, mz, phi

    # 1st stage, V_MTJ_1 and I_SOT for t1
    for i in range(sim_startup-1, sim_mid1):

        R_MTJ = R[i]

        mz, phi_temp, theta_temp = switching(
            V_MTJ_1, I_SOT_1, R_MTJ, theta, phi, ESTT, ESOT, VNV=VNV, NON=NON, R_SOT_FL_DL=0)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ_1, mz)

        V[i] = V_MTJ_1
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    # 2nd stage, V_MTJ_2 and 0 for next t2
    for i in range(sim_mid1, sim_mid2):

        R_MTJ = R[i]

        mz, phi_temp, theta_temp = switching(
            V_MTJ_2, I_SOT_2, R_MTJ, theta, phi, ESTT, ESOT, VNV=VNV, NON=NON, R_SOT_FL_DL=0)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(V_MTJ_2, mz)

        V[i] = V_MTJ_2
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    # 3rd stage, 
    for i in range(sim_mid2, sim_end):
        #no V_MTJ or I_SOT
        R_MTJ = R[i]

        mz, phi_temp, theta_temp = switching(
            0, 0, R_MTJ, theta, phi, ESTT, ESOT, VNV=VNV, NON=NON, R_SOT_FL_DL=0)
        phi, theta = phi_temp, theta_temp
        R_MTJ = tmr(0, mz)

        V[i] = 0
        # for next iteration
        M_z[i+1] = mz
        Theta[i+1] = theta
        Phi[i+1] = phi
        R[i+1] = R_MTJ

    return time*t_step, M_z


def paper_proposed_main_switch():
    """
    To reproduce paper in section III D Figure 6b
    VCMA-Enabled -> VNV = 1

    H_EX = -50 Oe = H_DL_SOT
    -> I_SOT =  (2*e*u0*Ms*tf*A2*(-50*1000/(4*pi)))/(h_bar*theta_SH)
    V_MTJ_1 = 1.4937
    V_MTJ_2 = -1
    From 0 to 25 ns
    """

    VNV = 1
    NON = 0
    V_MTJ_1, V_MTJ_2 = 1.4937, -1
    I_SOT = (2*e*u0*Ms*tf*A2*(-50*1000/(4*pi)))/(h_bar*theta_SH)
    I_SOT_1, I_SOT_2 = I_SOT, 0
    time_index = [  # (t1, t2)
        (25e-9, 0), (1.4e-9, 1.6e-9), (1.8e-9, 1.2e-9), (2.2e-9, 0.8e-9)
    ]

    for t1, t2 in time_index:
        time_plot, M_z = paper_proposed_main_switch_helper(
            t1, t2, V_MTJ_1, V_MTJ_2, I_SOT_1, I_SOT_2)
        tick_spacing = 5e-9
        # mz vs time
        plt.plot(time_plot, M_z, label='t1 = {}s, t2 = {}s'.format(t1, t2))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    plt.xlabel('time(s)')
    plt.ylabel('mz')
    plt.tight_layout()
    plt.grid()
    plt.legend(loc='center right',
               title='H_EX = -50 Oe, \nI_SOT = -6.2619uA\nV_MTJ_1 = 1.4937V\nV_MTJ_2 = -1V')
    plt.gcf().canvas.set_window_title('paper_proposed_main_switch')
    # plt.show()
    plt.savefig('result/paper_proposed_main_switch.png')
    plt.close()

def paper_proposed_switch_error_rate():
    """
    Find the switch error rate

    uses 300 iterations for every t1 value
    To reproduce section III D Figure 6c for SER

    t1+t2 = 3e-9 s
    t1 from 1.3e-9, 1.4e-9, 1.5e-9, 1.6e-9, 1.7e-9, 1.8e-9, 1.9e-9

    Warning: might take around 70 minutes
    """
    iterations_num = 300

    VNV = 1
    NON = 1 #thermal noise enabled
    V_MTJ_1, V_MTJ_2 = 1.4937, -1
    I_SOT = (2*e*u0*Ms*tf*A2*(-50*1000/(4*pi)))/(h_bar*theta_SH)
    I_SOT_1, I_SOT_2 = I_SOT, 0
    
    t1_matrix = [1.3e-9, 1.4e-9, 1.5e-9, 1.6e-9, 1.7e-9, 1.8e-9, 1.9e-9]
    SER_avg_matrix = []
    M_z_avg_matrix = []

    for t1 in t1_matrix:
        start_time = t.time()
        error_sum = 0
        M_z_sum = 0
        for k in range(iterations_num):
            time_plot, M_z = paper_proposed_main_switch_helper(
                t1, 3e-9-t1, V_MTJ_1, V_MTJ_2, I_SOT_1, I_SOT_2, NON=1, VNV=1)
            M_z_sum += M_z[int(t1/t_step)]
            if (M_z[25000] - (1.0) > 1e-1 or (1.0-M_z[25000] > 1e-1)):  # switching failure at final
                error_sum += 1
        
        error_avg = error_sum/iterations_num
        SER_avg_matrix.append(error_avg)
        M_z_avg = M_z_sum/iterations_num
        M_z_avg_matrix.append(M_z_avg)
        print('t1 = {}s, t2 = {}s, Elapsed = {}s'.format(t1, 3e-9-t1, t.time()-start_time))
    
    fig, axs = plt.subplots(2)
    axs[0].plot(t1_matrix, SER_avg_matrix, marker = '^')  # SER vs time
    axs[0].set(xlabel='t1(s)', ylabel='SER')
    axs[1].plot(t1_matrix, M_z_avg_matrix, marker='s')  # M_z_final vs time
    axs[1].set(xlabel='t1(s)', ylabel='mz')
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    
    plt.gcf().canvas.set_window_title('paper_proposed_switch_error_rate, t1+t2 = 3ns')
    plt.savefig('result/paper_proposed_switch_error_rate.png')
    plt.close()


if __name__ == '__main__':
    ################### reproduce manual ###################

    manual_main_1()
    manual_main_2()

    ################### reproduce paper ####################

    # paper_main_switch_without_VCMA()
    # paper_main_switch_without_VCMA_with_noise()

    # Caveat: takes around 45 min for 6@1000 iterations
    # paper_switch_error_withoutVCMA_withNoise()

    # paper_main_switch_with_VCMA_different_ISOT()
    # paper_main_switch_with_VCMA_different_V_MTJ()

    # paper_proposed_main_switch()
    # paper_proposed_switch_error_rate()