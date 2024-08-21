import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import mplcursors
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

def mahalanobis_distance(track, report, cov_inv):
    delta = track - report
    try:
        temp = np.dot(np.dot(delta.T, cov_inv), delta)
        if np.any(np.isnan(temp)) or np.any(temp < 0):
            print("Warning: Invalid distance calculation.")
        distance = np.sqrt(temp)
    except np.linalg.LinAlgError:
        print("Error: Covariance matrix inversion failed.")
        distance = np.nan
    return distance

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = chi2.ppf(0.95, 3)  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[:3] = [x, y, z]
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt
            self.Sf[3:] = [self.vx, self.vy, self.vz]
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q *= self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan2(z[i], np.sqrt(x[i]**2 + y[i]**2)) * 180 / np.pi)
        az.append(math.atan2(y[i], x[i]) * 180 / np.pi)

        if x[i] > 0.0:                
            az[i] = 90 - az[i]
        else:
            az[i] = 270 - az[i]
        
        if az[i] < 0.0:
            az[i] += 360
        
        if az[i] > 360:
            az[i] -= 360

    return r, az, el

# Function to generate hypotheses for clusters
def generate_hypotheses(tracks, reports):
    hypotheses = []
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            hypotheses.append([(i, j)])
    return hypotheses

# Function to calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                delta = reports[report_idx] - tracks[track_idx]
                distance = np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))
                prob *= np.exp(-0.5 * distance**2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize
    return probabilities

# Function to get association weights
def get_association_weights(hypotheses, probabilities):
    num_tracks = len(hypotheses[0])
    association_weights = [[] for _ in range(num_tracks)]
    
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                association_weights[track_idx].append((report_idx, prob))
    
    for track_weights in association_weights:
        track_weights.sort(key=lambda x: x[0])  # Sort by report index
        report_probs = {}
        for report_idx, prob in track_weights:
            if report_idx not in report_probs:
                report_probs[report_idx] = prob
            else:
                report_probs[report_idx] += prob
        track_weights[:] = [(report_idx, prob) for report_idx, prob in report_probs.items()]
    
    return association_weights

# Function to calculate joint probabilities
def calculate_joint_probabilities(hypotheses, probabilities, association_weights):
    joint_probabilities = []
    for hypothesis, prob in zip(hypotheses, probabilities):
        joint_prob = prob
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                weight = next(w for r, w in association_weights[track_idx] if r == report_idx)
                joint_prob *= weight
        joint_probabilities.append(joint_prob)
    return joint_probabilities

def find_max_associations(hypotheses, probabilities):
    max_associations = [-1] * len(hypotheses[0])
    max_probs = [-np.inf] * len(hypotheses[0])  # Initialize with -inf for comparison
    
    for hypothesis, prob_array in zip(hypotheses, probabilities):
        prob = np.asscalar(np.array(prob_array)) if isinstance(prob_array, np.ndarray) else prob_array
        
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                if prob > max_probs[report_idx]:
                    max_probs[report_idx] = prob
                    max_associations[report_idx] = track_idx
    
    return max_associations, max_probs


def main():
    file_path = "ttk_84_2.csv"
    measurements = read_measurements_from_csv(file_path)
    measurement_groups = form_measurement_groups(measurements)
    
    tracks = [CVFilter() for _ in range(1)]  # Track ID is 1

    for group in measurement_groups:
        reports = np.array([measurement[:3] for measurement in group])
        report_times = np.array([measurement[3] for measurement in group])
        
        for track in tracks:
            track.predict_step(report_times[0])
        
        clusters = [(list(range(len(tracks))), list(range(len(reports))))]  # Assume one cluster for simplicity
        
        for track_idxs, report_idxs in clusters:
            cluster_tracks = [tracks[i].Sf[:3] for i in track_idxs]
            cluster_reports = reports[report_idxs]
            cov_inv = np.linalg.inv(tracks[0].Pp[:3, :3])  # Assuming all tracks have the same covariance matrix
            
            hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
            probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
            association_weights = get_association_weights(hypotheses, probabilities)
            joint_probabilities = calculate_joint_probabilities(hypotheses, probabilities, association_weights)
            max_associations, max_probs = find_max_associations(hypotheses, probabilities)
            
            print("Hypotheses:")
            print("Tracks/Reports:", ["t" + str(i+1) for i in track_idxs])
            for hypothesis, prob, joint_prob in zip(hypotheses, probabilities, joint_probabilities):
                formatted_hypothesis = ["r" + str(report_idxs[r]+1) if r != -1 else "0" for _, r in hypothesis]

                # Ensure prob and joint_prob are scalars
                prob_scalar = np.asscalar(np.array(prob)) if isinstance(prob, np.ndarray) else prob
                joint_prob_scalar = np.asscalar(np.array(joint_prob)) if isinstance(joint_prob, np.ndarray) else joint_prob

                (f"Hypothesis: {formatted_hypothesis}, Probability: {prob_scalar:.4f}, Joint Probability: {joint_prob_scalar:.4f}")
            for track_idx, weights in enumerate(association_weights):
                max_weight = 0.0
                report_id = -1
                for report_idx, weight in weights:
                    if weight > max_weight:
                        max_weight = weight
                        report_id = report_idx + 1
                print(f"Most likely association for Report r{report_id}: Track t{track_idxs[track_idx]+1}, weight: {max_weight:.4f}")

        for report, assoc in zip(reports, max_associations):
            if assoc != -1:
                tracks[assoc].update_step(report)

    # After processing, convert results to spherical coordinates
    filtered_values = np.array([track.Sf[:3].flatten() for track in tracks])
    r, az, el = cart2sph2(filtered_values[:, 0], filtered_values[:, 1], filtered_values[:, 2], filtered_values)

    # Plotting the results
    plt.figure()
    plt.scatter(r, az, c=el, cmap='viridis')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.colorbar(label='Elevation')
    plt.title('Filtered Measurements in Spherical Coordinates')
    mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    main()
