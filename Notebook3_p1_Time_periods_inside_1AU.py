import numpy as np
import rebound
from tqdm import tqdm
from rebound.tools import M_to_f
from concurrent.futures import ProcessPoolExecutor, as_completed

def rotation_matrix_to_target(v1, target):
    # Normalize input vectors
    v1 = np.array(v1, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    v1 = v1 / np.linalg.norm(v1)
    target = target / np.linalg.norm(target)
    
    # Compute cross product and sine of angle
    axis = np.cross(v1, target)
    sin_theta = np.linalg.norm(axis)
    
    # Compute dot product and cosine of angle
    cos_theta = np.dot(v1, target)
    
    # If the vectors are already aligned
    if sin_theta == 0 and cos_theta > 0:
        return np.eye(3)
    
    # If the vectors are opposite
    if sin_theta == 0 and cos_theta < 0:
        # Return a rotation of 180 degrees around an orthogonal vector
        # Find an orthogonal vector
        orthogonal = np.array([1, 0, 0]) if abs(v1[0]) < abs(v1[1]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + K + K @ K
    
    # Normalize the axis
    axis = axis / sin_theta
    
    # Compute the skew-symmetric cross-product matrix of the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Compute the rotation matrix
    R = np.eye(3) + np.sin(np.arcsin(sin_theta)) * K + (1 - cos_theta) * (K @ K)
    return R

def calculate_1AU_true_anomaly(a, e):
    f_1 = np.arccos((a * (1. - e**2) - 1.) / (e * 1.))
    f_2 = 2 * np.pi - np.arccos((a * (1. - e**2) - 1.) / (e * 1.))
    return f_1, f_2

def calculate_Mean_anomaly(e, f):
    def M_ell(e, f):
        M_ell_ = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2)) - e * np.sqrt(1 - e**2) * np.sin(f) / (1 + e * np.cos(f))
        return M_ell_
    
    def M_hyp(e, f):
        M_hyp_ = (e * np.sqrt(e**2 - 1) * np.sin(f)) / (1 + e * np.cos(f)) - np.log((np.sqrt(e + 1) + np.sqrt(e - 1) * np.tan(f / 2)) / (np.sqrt(e + 1) - np.sqrt(e - 1) * np.tan(f / 2)))
        return M_hyp_
    
    Mean_anomaly = np.zeros(len(f))
    for i in range(len(f)):
        if e[i] < 1:
            Mean_anomaly[i] = M_ell(e[i], f[i])
            if Mean_anomaly[i] < 0:
                Mean_anomaly[i] = 2 * np.pi + Mean_anomaly[i]
        else:
            Mean_anomaly[i] = M_hyp(e[i], f[i])
    return Mean_anomaly

def falling_time(time_impact, Mean_anomaly, a, e, G, M, f_1, f_2):
    ang_vel = np.sqrt(G * M / np.abs(a)**3)
    
    M_1 = calculate_Mean_anomaly(e, f_1)
    M_2 = calculate_Mean_anomaly(e, f_2)
    
    time_ell_1 = time_impact + abs(Mean_anomaly - M_2) / ang_vel
    time_ell_2 = time_impact + abs(2 * np.pi - Mean_anomaly + M_1) / ang_vel
    falling_times_ell = np.empty((len(time_ell_1), 2)) 
    falling_times_ell[:, 0] = time_ell_1 
    falling_times_ell[:, 1] = time_ell_2  

    # falling_times_ell = np.vstack((time_ell_1, time_ell_2))
    
    time_hyp_1 = np.where(Mean_anomaly < 0, time_impact + abs(Mean_anomaly - M_2) / ang_vel, np.inf)
    time_hyp_2 = np.where(Mean_anomaly < 0, time_impact + abs(Mean_anomaly - M_1) / ang_vel, np.inf)
    falling_times_hyp = np.empty((len(time_hyp_1), 2))
    falling_times_hyp[:, 0] = time_hyp_1
    falling_times_hyp[:, 1] = time_hyp_2
    
    # falling_times_hyp = np.vstack((time_hyp_1, time_hyp_2))
    
    falling_times = np.where(a.T[:, np.newaxis] < 0, falling_times_hyp, falling_times_ell)

    time_points = falling_times

    valid = np.all(np.isfinite(time_points), axis=1)
    time_points_filtered = time_points[valid, :]

    return time_points_filtered 


seeds = np.linspace(0, 10000, 10001)

def process_seed(seed):
    
    seed = int(seed)
    np.random.seed(seed)
    
    # Set the units
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    G = sim.G

    # The Sun
    M = 1 # Msun

    # HD 7977
    v0 = 26.4485 # km/s
    v_vector = np.array([11.67373463, 13.27281649, 19.67970122]) # km/s, calculated from the radial velocity and proper motion
    km_to_au = 1 / 149597870.7
    s_to_year = 31557600
    conversion_factor = km_to_au * s_to_year
    vp = v0 * conversion_factor
    Mp = 1.07 # Msun
    bp = 2300  # AU, also has other data says 37380.51 AU
    time_p = 2.47e6 # yr, closest approach time

    N_sim_points = 1000000
    
    # Set the inner Oort Cloud
    a = 10**np.random.uniform(np.log10(2000), np.log10(20000), N_sim_points)
    e2 = np.random.uniform(0, 1, N_sim_points)
    e = np.sqrt(e2)
    q = a * (1 - e)

    # filter with q > 10 AU
    initial_filter = q > 10
    a = a[initial_filter]
    e = e[initial_filter]
    q = q[initial_filter]
    N_sim_points = len(a)

    phi = np.random.uniform(0, 2 * np.pi, N_sim_points) # orientation of the ellipse, also the argument of pericenter (omega)

    M_ = np.random.uniform(0, 2 * np.pi, N_sim_points) # mean anomaly
    f = np.array([M_to_f(e[i], M_[i]) for i in range(N_sim_points)])  # true anomaly

    alpha = phi + f  # angle from the x-axis to the position vector
    Omega = np.random.uniform(0, 2 * np.pi, N_sim_points)  # longitude of the ascending node

    inc = np.where(a > 5000, np.arccos(np.random.uniform(-1, 1, N_sim_points)), 
            np.random.normal(np.pi/6, np.pi/18, N_sim_points))  # inclination

    c = a * e  # distance from the center to the focus
    b = np.sqrt(a**2 - c**2)  # semi-minor axis

    r = a * (1 - e**2) / (1 + e * np.cos(f))

    tan_theta_phi = -b**2 / a**2 * (c + r * np.cos(f)) / (r * np.sin(f))
    theta = np.where(np.array(f) < np.pi, np.arctan(tan_theta_phi) + phi + np.pi, np.arctan(tan_theta_phi) + phi)
    v_in0 = np.sqrt(G * M * (2 / r - 1 / a))

    Zero = np.zeros(N_sim_points)
    r_0 = np.array([r * np.cos(alpha), r * np.sin(alpha), Zero])
    v_0 = np.array([v_in0 * np.cos(theta), v_in0 * np.sin(theta), Zero])

    # semi-minor axis : (a1,a2,a3)
    a1 = -np.sin(phi) 
    a2 = np.cos(phi)
    a3 = np.zeros(N_sim_points)
    aa = np.array([a1, a2, a3])

    # rotation matrix
    MM = np.array([
        [np.cos(inc) + (1 - np.cos(inc)) * a1**2, (1 - np.cos(inc)) * a1 * a2 - np.sin(inc) * a3, (1 - np.cos(inc)) * a1 * a3 + np.sin(inc) * a2],
        [(1 - np.cos(inc)) * a1 * a2 + np.sin(inc) * a3, np.cos(inc) + (1 - np.cos(inc)) * a2**2, (1 - np.cos(inc)) * a2 * a3 - np.sin(inc) * a1],
        [(1 - np.cos(inc)) * a1 * a3 - np.sin(inc) * a2, (1 - np.cos(inc)) * a2 * a3 + np.sin(inc) * a1, np.cos(inc) + (1 - np.cos(inc)) * a3**2]
    ])

    r_1 = np.einsum('ijk,jk->ik', MM, r_0)
    v_1 = np.einsum('ijk,jk->ik', MM, v_0)

    # normal vector
    dd = np.cross(r_1, v_1, axis=0)
    dd = dd / np.sqrt(dd[0]**2 + dd[1]**2 + dd[2]**2)
    d1 = dd[0]
    d2 = dd[1]
    d3 = dd[2]

    NN = np.array([
        [np.cos(Omega) + (1 - np.cos(Omega)) * d1**2, (1 - np.cos(Omega)) * d1 * d2 - np.sin(Omega) * d3, (1 - np.cos(Omega)) * d1 * d3 + np.sin(Omega) * d2],
        [(1 - np.cos(Omega)) * d1 * d2 + np.sin(Omega) * d3, np.cos(Omega) + (1 - np.cos(Omega)) * d2**2, (1 - np.cos(Omega)) * d2 * d3 - np.sin(Omega) * d1],
        [(1 - np.cos(Omega)) * d1 * d3 - np.sin(Omega) * d2, (1 - np.cos(Omega)) * d2 * d3 + np.sin(Omega) * d1, np.cos(Omega) + (1 - np.cos(Omega)) * d3**2]
    ])

    r_1 = np.einsum('ijk,jk->ik', NN, r_1)
    v_1 = np.einsum('ijk,jk->ik', NN, v_1)

    time_be_impact = -time_p + r_1[2] / vp

    r_in = np.sqrt(r_1[0]**2 + r_1[1]**2 + r_1[2]**2)
    v_in = np.sqrt(v_1[0]**2 + v_1[1]**2 + v_1[2]**2)

    E = -G * M / r_in + 0.5 * v_in**2

    delta_v_x = 2 * G * Mp * (bp - r_1[0]) / vp / ((bp - r_1[0])**2 + (r_1[2])**2)
    delta_v_y = np.zeros(N_sim_points)
    delta_v_z = - 2 * G * Mp * r_1[2] / vp / ((bp - r_1[0])**2 + (r_1[2])**2)

    delta_vx_sun = 2 * G * Mp / vp / bp
    delta_vy_sun = 0    
    delta_vz_sun = 0

    delta_v_x = delta_v_x - delta_vx_sun
    delta_v_y = delta_v_y - delta_vy_sun
    delta_v_z = delta_v_z - delta_vz_sun

    RR = rotation_matrix_to_target([0, 1, 0], v_vector)
    delta_v = np.array([delta_v_x, delta_v_y, delta_v_z])
    delta_v = RR @ delta_v  

    v_after = v_1 + delta_v

    E_after = -G * M / r_in + 0.5 * v_after[0]**2 + 0.5 * v_after[1]**2 + 0.5 * v_after[2]**2

    a_after = -G * M / 2 / E_after

    ang_mom_after = np.cross(r_1, v_after, axis=0)
    ang_mom_after = np.sqrt(ang_mom_after[0]**2 + ang_mom_after[1]**2 + ang_mom_after[2]**2)

    e_after = np.sqrt(1 - ang_mom_after**2 / (G * M * a_after))

    f_after = np.zeros(N_sim_points)

    for i in range(N_sim_points):
        # Excluding the effects of numerical instability
        if ((a_after[i] * (1 - e_after[i]**2) - r_in[i]) / (e_after[i] * r_in[i])) > 1:
            f_after[i] = 0
        elif ((a_after[i] * (1 - e_after[i]**2) - r_in[i]) / (e_after[i] * r_in[i])) < -1:
            f_after[i] = np.pi
        else:
            if r_1[0, i] * v_after[0, i] + r_1[1, i] * v_after[1, i] + r_1[2, i] * v_after[2, i] > 0:
                f_after[i] = np.arccos((a_after[i] * (1 - e_after[i]**2) - r_in[i]) / (e_after[i] * r_in[i]))
            else:
                f_after[i] = 2 * np.pi - np.arccos((a_after[i] * (1 - e_after[i]**2) - r_in[i]) / (e_after[i] * r_in[i]))

    perihelion = ang_mom_after**2 / ((e_after + 1) * G * M)
    
    Filter_1AU = perihelion < 1
    a_after = a_after[Filter_1AU]
    e_after = e_after[Filter_1AU]
    f_after = f_after[Filter_1AU]
    ang_mom_after = ang_mom_after[Filter_1AU]
    time_be_impact_after = time_be_impact[Filter_1AU]
    q_after = perihelion[Filter_1AU]

    f_1_after, f_2_after = calculate_1AU_true_anomaly(a_after, e_after)
    Mean_anomaly_after = calculate_Mean_anomaly(e_after, f_after)
    time_points_after = falling_time(time_be_impact_after, Mean_anomaly_after, a_after, e_after, G, M, f_1_after, f_2_after)
    
    # Outer Oort Cloud (steady state)
    N_oort = int(N_sim_points/10)
    a_oort = 10**np.random.uniform(np.log10(20000), np.log10(100000), N_oort)
    e2_oort = np.random.uniform(0, 1, N_oort)
    e_oort = np.sqrt(e2_oort)
    q_oort = a_oort * (1 - e_oort)

    Filter_1AU_oort = q_oort < 1
    a_oort = a_oort[Filter_1AU_oort]
    e_oort = e_oort[Filter_1AU_oort]
    q_oort = q_oort[Filter_1AU_oort]
    N_oort = len(a_oort)

    f_oort1, f_oort2 = calculate_1AU_true_anomaly(a_oort, e_oort)
    M_oort = np.random.uniform(0, 2 * np.pi, N_oort)
    time_points_oort = falling_time(-time_p-1e6, M_oort, a_oort, e_oort, G, M, f_oort1, f_oort2)

    return time_points_after, time_points_oort

if __name__ == '__main__':

    times_inner = []
    times_outer = []

    with ProcessPoolExecutor(max_workers=10) as executor:   # change the number of workers here
        futures = [executor.submit(process_seed, seed) for seed in seeds]
        for future in tqdm(as_completed(futures), total=len(seeds)):
                time_points_after, time_points_oort = future.result()
                times_inner.append(time_points_after)
                times_outer.append(time_points_oort)
    
    times_inner = np.concatenate(times_inner, axis=0)
    times_outer = np.concatenate(times_outer, axis=0)
    
    np.save('times_inner.npy', times_inner)
    np.save('times_outer.npy', times_outer)
    


    
    

