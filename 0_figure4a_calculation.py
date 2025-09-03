import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed  

def coth(x):
    return np.cosh(x) / np.sinh(x)


#STO
Q_0 = 1e12
T = 50   
alpha_1 = 405e5* (coth(54/T)-coth(54/30))
beta_1 = 1.32e29/(Q_0**2)* (coth(145/T)-coth(145/105))  

alpha_11 = 2.8992745136959114 * 1e9
alpha_12 = 7.766002866404304 * 1e9

beta_11 = 1.688325922985312 * 1e50/(Q_0**4)
beta_12 = 3.87895181502471 * 1e50/(Q_0**4)

t_11 = -1.9024136562127822 * 1e29/(Q_0**2)
t_12 = -1.0142477173481538 * 1e29/(Q_0**2)
t_44 = 5.865354330708662 * 1e29/(Q_0**2)


c_11 = 3.36 * 1e11
c_12 = 1.07 * 1e11
c_44 = 1.27 * 1e11


Q_11 = 0.05357284636760619
Q_12 = -0.015422786820166734
Q_44 = 0.004724409448818898

Lambda_11 = 8.820166732830488 * 1e18/(Q_0**2)
Lambda_12 = -7.773719730051609 * 1e18/(Q_0**2)
Lambda_44 = -4.52755905511811 * 1e18/(Q_0**2)





def ff_LGD(P1, P2, P3, q1, q2, q3):
    return (alpha_1 * (P1**2 + P2**2 + P3**2) +
            alpha_11 * (P1**4 + P2**4 + P3**4) +
            alpha_12 * (P1**2 * P2**2 + P1**2 * P3**2 + P2**2 * P3**2) +
            beta_1 * (q1**2 + q2**2 + q3**2) +
            beta_11 * (q1**4 + q2**4 + q3**4) +
            beta_12 * (q1**2 * q2**2 + q1**2 * q3**2 + q2**2 * q3**2) -
            t_11 * (P1**2 * q1**2 + P2**2 * q2**2 + P3**2 * q3**2) - 
            t_12 * (P1**2 * (q2**2 + q3**2) + P2**2 * (q1**2 + q3**2) + P3**2 * (q1**2 + q2**2)) -
            t_44 * (P1 * P2 * q1 * q2 + P2 * P3 * q2 * q3 + P1 * P3 * q1 * q3)
            )


def epsilon0_xx(P1, P2, P3, q1, q2, q3):
    return Q_11 * P1**2 + Q_12 * (P2**2 + P3**2) + Lambda_11 * q1**2 + Lambda_12 * (q2**2 + q3**2)

def epsilon0_yy(P1, P2, P3, q1, q2, q3):
    return Q_11 * P2**2 + Q_12 * (P1**2 + P3**2) + Lambda_11 * q2**2 + Lambda_12 * (q1**2 + q3**2)

def epsilon0_zz(P1, P2, P3, q1, q2, q3):
    return Q_11 * P3**2 + Q_12 * (P1**2 + P2**2) + Lambda_11 * q3**2 + Lambda_12 * (q1**2 + q2**2)

def epsilon0_yz(P1, P2, P3, q1, q2, q3):
    return Q_44 * P2*P3 + Lambda_44 * q2*q3

def epsilon0_xz(P1, P2, P3, q1, q2, q3):
    return Q_44 * P1*P3 + Lambda_44 * q1*q3

def epsilon0_xy(P1, P2, P3, q1, q2, q3):
    return Q_44 * P1*P2 + Lambda_44 * q1*q2

def epsilon0_app(P1, P2, P3, q1, q2, q3, theta):
    e0xx = epsilon0_xx(P1, P2, P3, q1, q2, q3)
    e0yy = epsilon0_yy(P1, P2, P3, q1, q2, q3)
    e0xy = epsilon0_xy(P1, P2, P3, q1, q2, q3)

    return np.cos(theta)**2 * e0xx + np.sin(theta)**2 * e0yy -2 * np.cos(theta)*np.sin(theta) * e0xy

def epsilon_xx(P1, P2, P3, q1, q2, q3, epsilon_app_new, theta):

    epsilon_app = (epsilon0_app(P1, P2, P3, q1, q2, q3, theta)+1)* (epsilon_app_new+1)-1
    
    numerator = (
        epsilon0_xx(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)**2 * (c_11 - c_12) * (c_11 + 2 * c_12)
        + 2 * (
            2 * (2 * epsilon0_xy(P1, P2, P3, q1, q2, q3) * np.cos(theta)**3 * np.sin(theta) + epsilon0_xx(P1, P2, P3, q1, q2, q3) * np.sin(theta)**4 
                 + np.cos(theta)**2 * (epsilon_app - epsilon0_yy(P1, P2, P3, q1, q2, q3) * np.sin(theta)**2)) * c_11
            + np.cos(2 * theta) * (
                -epsilon0_xx(P1, P2, P3, q1, q2, q3) - epsilon0_yy(P1, P2, P3, q1, q2, q3) + 2 * epsilon_app 
                + (epsilon0_xx(P1, P2, P3, q1, q2, q3) + epsilon0_yy(P1, P2, P3, q1, q2, q3)) * np.cos(2 * theta) 
                + 2 * epsilon0_xy(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)
            ) * c_12
        ) * c_44
    )

    # 计算分母
    denominator = (
        np.sin(2 * theta)**2 * c_11**2
        + c_12 * ((-1 + np.cos(4 * theta)) * c_12 + 4 * np.cos(2 * theta)**2 * c_44)
        + c_11 * (np.sin(2 * theta)**2 * c_12 + (3 + np.cos(4 * theta)) * c_44)
    )

    return numerator / denominator


def epsilon_yy(P1, P2, P3, q1, q2, q3, epsilon_app_new, theta):

    epsilon_app = (epsilon0_app(P1, P2, P3, q1, q2, q3, theta)+1)* (epsilon_app_new+1)-1

    numerator = (
        epsilon0_yy(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)**2 * (c_11 - c_12) * (c_11 + 2 * c_12)
        + (
            (4 * epsilon0_yy(P1, P2, P3, q1, q2, q3) * np.cos(theta)**4 
             + 4 * epsilon_app * np.sin(theta)**2 
             + 8 * epsilon0_xy(P1, P2, P3, q1, q2, q3) * np.cos(theta) * np.sin(theta)**3 
             - epsilon0_xx(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)**2) * c_11
            + 2 * np.cos(2 * theta) * (
                epsilon0_xx(P1, P2, P3, q1, q2, q3) + epsilon0_yy(P1, P2, P3, q1, q2, q3) 
                - 2 * epsilon_app 
                + (epsilon0_xx(P1, P2, P3, q1, q2, q3) + epsilon0_yy(P1, P2, P3, q1, q2, q3)) * np.cos(2 * theta) 
                - 2 * epsilon0_xy(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)
            ) * c_12
        ) * c_44
    )


    denominator = (
        np.sin(2 * theta)**2 * c_11**2
        + c_12 * ((-1 + np.cos(4 * theta)) * c_12 + 4 * np.cos(2 * theta)**2 * c_44)
        + c_11 * (np.sin(2 * theta)**2 * c_12 + (3 + np.cos(4 * theta)) * c_44)
    )

    return numerator / denominator


def epsilon_zz(P1, P2, P3, q1, q2, q3, epsilon_app_new, theta):

    epsilon_app = (epsilon0_app(P1, P2, P3, q1, q2, q3, theta)+1)* (epsilon_app_new+1)-1

    numerator = (
        epsilon0_zz(P1, P2, P3, q1, q2, q3) * (-1 + np.cos(4 * theta)) * (c_11 - c_12) * (c_11 + 2 * c_12)
        - 2 * (
            epsilon0_zz(P1, P2, P3, q1, q2, q3) * (3 + np.cos(4 * theta)) * c_11
            + 2 * (
                epsilon0_xx(P1, P2, P3, q1, q2, q3) + epsilon0_yy(P1, P2, P3, q1, q2, q3) + epsilon0_zz(P1, P2, P3, q1, q2, q3) 
                - 2 * epsilon_app 
                + (epsilon0_xx(P1, P2, P3, q1, q2, q3) - epsilon0_yy(P1, P2, P3, q1, q2, q3)) * np.cos(2 * theta)
                + epsilon0_zz(P1, P2, P3, q1, q2, q3) * np.cos(4 * theta)
                - 2 * epsilon0_xy(P1, P2, P3, q1, q2, q3) * np.sin(2 * theta)
            ) * c_12
        ) * c_44
    )


    denominator = (
        (-1 + np.cos(4 * theta)) * (c_11 - c_12) * (c_11 + 2 * c_12)
        - 2 * ((3 + np.cos(4 * theta)) * c_11 + 4 * np.cos(2 * theta)**2 * c_12) * c_44
    )

    return numerator / denominator

def epsilon_yz(P1, P2, P3, q1, q2, q3, epsilon_app, theta):
    return epsilon0_yz(P1, P2, P3, q1, q2, q3)

def epsilon_xz(P1, P2, P3, q1, q2, q3, epsilon_app, theta):
    return epsilon0_xz(P1, P2, P3, q1, q2, q3)

def epsilon_xy(P1, P2, P3, q1, q2, q3, epsilon_app_new, theta):

    epsilon_app = (epsilon0_app(P1, P2, P3, q1, q2, q3, theta)+1)* (epsilon_app_new+1)-1

    numerator = (
        0.5 * (epsilon0_xx(P1, P2, P3, q1, q2, q3) + epsilon0_yy(P1, P2, P3, q1, q2, q3) 
               - 2 * epsilon_app + (epsilon0_xx(P1, P2, P3, q1, q2, q3) - epsilon0_yy(P1, P2, P3, q1, q2, q3)) * np.cos(2 * theta))
        * np.sin(2 * theta) * (c_11 - c_12) * (c_11 + 2 * c_12)
        + epsilon0_xy(P1, P2, P3, q1, q2, q3) * ((3 + np.cos(4 * theta)) * c_11 + 4 * np.cos(2 * theta)**2 * c_12) * c_44
    )


    denominator = (
        np.sin(2 * theta)**2 * c_11**2
        + c_12 * ((-1 + np.cos(4 * theta)) * c_12 + 4 * np.cos(2 * theta)**2 * c_44)
        + c_11 * (np.sin(2 * theta)**2 * c_12 + (3 + np.cos(4 * theta)) * c_44)
    )

    return numerator / denominator



def ff_Elas(P1, P2, P3, q1, q2, q3, epsilon_app, theta):

    ee_xx = epsilon_xx(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_xx(P1, P2, P3, q1, q2, q3)
    ee_yy = epsilon_yy(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_yy(P1, P2, P3, q1, q2, q3)
    ee_zz = epsilon_zz(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_zz(P1, P2, P3, q1, q2, q3)
    ee_yz = epsilon_yz(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_yz(P1, P2, P3, q1, q2, q3)
    ee_xz = epsilon_xz(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_xz(P1, P2, P3, q1, q2, q3)
    ee_xy = epsilon_xy(P1, P2, P3, q1, q2, q3, epsilon_app, theta) - epsilon0_xy(P1, P2, P3, q1, q2, q3)


    elas_energy = (
        0.5 * c_11 * (ee_xx**2 + ee_yy**2 + ee_zz**2) +
        c_12 * (ee_xx * ee_yy + ee_xx * ee_zz + ee_yy * ee_zz) +
        2*c_44 * (ee_xy**2 + ee_xz**2 + ee_yz**2)
    )

    return elas_energy


def ff_total(P, epsilon_app, theta):
    P1, P2, P3, q1, q2, q3 = P
    return ff_LGD(P1, P2, P3, q1, q2, q3)+ff_Elas(P1, P2, P3, q1, q2, q3, epsilon_app, theta)







#methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP']
methods = ['Powell', 'L-BFGS-B', 'TNC', 'SLSQP']


num_random_samples = 1000 


epsilon_app_range = np.arange(0, 0.02, 0.0001)
theta_range = np.linspace(0, np.pi / 2, 200)


results = []


def random_search(ff_total, epsilon_app, theta, bounds, num_samples=1000):
    best_energy = np.inf
    best_point = None
    for _ in range(num_samples):
        P = [np.random.uniform(low, high) for low, high in bounds]
        energy = ff_total(P, epsilon_app, theta)
        if energy < best_energy:
            best_energy = energy
            best_point = P
    return best_point, best_energy


def optimize_and_classify(epsilon_app, theta):
    initial_guesses = [
        [0, 1, 0.01, 1, 1, 1],
        [1, 0, 0.01, 1, 1, 1],
        [0.5, 0.5, 0.01, 1, 1, 1],
        [0.01, 0.5, 0.5, 1, 1, 1],
        [0.5, 0.01, 0.5, 1, 1, 1],
    ]
    best_result = None
    best_energy = np.inf

    for method in methods:
        for x0 in initial_guesses:
            try:
                res = minimize(
                    lambda P: ff_total(P, epsilon_app, theta),
                    x0=x0,
                    bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)],
                    method=method
                )

                if res.success and res.fun < best_energy:
                    best_energy = res.fun
                    best_result = res

            except Exception as e:
                print(f"Warning: Method {method} with initial guess {x0} failed with error: {e}")

    if best_result:
        P1, P2, P3, q1, q2, q3 = best_result.x


        delta = abs(ff_total([P1, P2, P3, q1, q2, q3], epsilon_app, theta) - ff_total([P2, P1, P3, q2, q1, q3], epsilon_app, theta))
        e_threshold = 1e2
        P_threshold = 1e-2
        q_threshold = 1e-1

        if P1 > P_threshold and P2 < P_threshold and P3 < P_threshold:
            phaseP = "Phase 1"       # P1 
        elif P1 < P_threshold and P2 > P_threshold and P3 < P_threshold:
            phaseP = "Phase 2"       # P2
        elif P1 < P_threshold and P2 < P_threshold and P3 > P_threshold:
            phaseP = "Phase 3"       # P3
        elif P1 > P_threshold and P2 > P_threshold and P3 < P_threshold:
            phaseP = "Phase 4"       # P1 and P2
        elif P1 > P_threshold and P2 < P_threshold and P3 > P_threshold:
            phaseP = "Phase 5"       # P1 and P3
        elif P1 < P_threshold and P2 > P_threshold and P3 > P_threshold:
            phaseP = "Phase 6"       # P2 and P3
        elif P1 > P_threshold and P2 > P_threshold and P3 > P_threshold:
            phaseP = "Phase 7"       # P1 and P2 and P3
        else:
            phaseP = "Unclassified"

        if q1 > q_threshold and q2 < q_threshold and q3 < P_threshold:
            phaseq = "Phase 1"       # q1 
        elif q1 < q_threshold and q2 > q_threshold and q3 < q_threshold:
            phaseq = "Phase 2"       # q2
        elif q1 < q_threshold and q2 < q_threshold and q3 > q_threshold:
            phaseq = "Phase 3"       # q3
        elif q1 > q_threshold and q2 > q_threshold and q3 < q_threshold:
            phaseq = "Phase 4"       # q1 and q2
        elif q1 > q_threshold and q2 < q_threshold and q3 > q_threshold:
            phaseq = "Phase 5"       # q1 and q3
        elif q1 < q_threshold and q2 > q_threshold and q3 > q_threshold:
            phaseq = "Phase 6"       # q2 and q3
        elif q1 > q_threshold and q2 > q_threshold and q3 > q_threshold:
            phaseq = "Phase 7"       # q1 and q2 and q3
        else:
            phaseq = "Unclassified"
        

        

        return [epsilon_app, theta, P1, P2, P3, q1, q2, q3, best_energy, phaseP, phaseq]
    else:
        return [epsilon_app, theta, None, None, None, None, None, None, None, None, "Unclassified"]


total_steps = len(epsilon_app_range) * len(theta_range)
current_step = 0

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(optimize_and_classify, epsilon_app, theta): (epsilon_app, theta) for epsilon_app in epsilon_app_range for theta in theta_range}
    
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        current_step += 1
        print(f"Completed step {current_step} / {total_steps}")


df = pd.DataFrame(results, columns=["epsilonApp", "theta", "P1", "P2", "P3", "q1", "q2", "q3", "Energy", "PhaseP", "Phaseq"])


df.to_csv("pto_phase_results_STO_50K.csv", index=False)

print("Results saved to 'sto_phase_results.csv'")