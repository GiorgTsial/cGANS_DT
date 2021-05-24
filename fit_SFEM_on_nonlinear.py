from BeamSFEM.Beam_PlainStress import Beam as beam_linear
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy


def KL_distance(real_samples, generated_samples, bandwidth, lower_plot_val, upper_plot_val):
    kd_real = KernelDensity(bandwidth=bandwidth)
    kd_fakes = KernelDensity(bandwidth=bandwidth)

    kd_real.fit(real_samples.reshape(-1, 1))
    kd_fakes.fit(generated_samples.reshape(-1, 1))

    x = np.linspace(lower_plot_val, upper_plot_val, 100).reshape(-1, 1)

    real_scores = np.exp(kd_real.score_samples(x))

    fake_scores = np.exp(kd_fakes.score_samples(x))

    kl_diverg = entropy(real_scores, fake_scores)

    return kl_diverg


training_keys = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0]
validation_keys = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0]
testing_keys = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0]

base_E = float(2 * 10 ** 9)
E_stdv = 0.2
h = 0.4
L = 5.0
rho = 7800
T = 30.0
t = 0.1
CorrLength = [3.0]
base_element_size = 0.1
zeta = 0.05
poisson = 0.3

Es = np.linspace(float(1 * 10 ** 9), float(2.5 * 10 ** 9), 15)
clens = np.linspace(2.0, 4.0, 20)
stdvs = np.linspace(0.1, 0.3, 20)


print(Es)
print(clens)
print(stdvs)

cwd = os.getcwd()
tip_disps_path = os.path.join(cwd, "data", "nonlin_tip_disps.csv")
all_nonlin_tip_disps = np.genfromtxt(tip_disps_path, delimiter=',')

nonlin_tip_disps_dict = {}
loads = np.linspace(1, 40, 40)
for load in loads:
    nonlin_tip_disps_dict[load] = all_nonlin_tip_disps[int(load)-1, :]

bandwidth = (np.max(all_nonlin_tip_disps) - np.min(all_nonlin_tip_disps)) / 20

lower_plot_val = np.min(all_nonlin_tip_disps) / 1.1
upper_plot_val = np.max(all_nonlin_tip_disps) * 1.1

print(bandwidth)

logs_file_path = os.path.join(cwd, "logs.txt")
with open(logs_file_path, "w") as fp:
    fp.write("E, Correlation Length, Stdv, KL Divergence\n")

best_KL_divergence = 1000000000000000000000000
best_beam = None
best_comb = (None, None, None)
for E in Es:
    for clen in clens:
        for stdv in stdvs:
            print("Testing E = ", E, ", Corr Length = ", clen, ", stdv = ", stdv)

            current_KL_divergence = 0
            for load in training_keys:
                beam = beam_linear(base_E=base_E, E_stdv=E_stdv, h=h, L=L, rho=rho, T=T, t=t,
                                   base_element_size=base_element_size, zeta=zeta, poisson=poisson,
                                   CorrLength=CorrLength)
                Us, CoVU = beam.StaticSSFEM(load=load)
                sfem_samples = beam.generate_static_samples_SSFEM(500, Us=Us)
                sfem_samples = sfem_samples[:, -1]
                current_KL_divergence += KL_distance(nonlin_tip_disps_dict[load], sfem_samples, bandwidth, lower_plot_val, upper_plot_val)

            with open(logs_file_path, "a") as fp:
                fp.write(str(E) + ", " + str(clen) + ", " + str(stdv) + ", " + str(current_KL_divergence) + "\n")

            if current_KL_divergence < best_KL_divergence:
                best_KL_divergence = current_KL_divergence
                best_beam = beam
                best_comb = (E, clen, stdv)

print("Best combination: ", best_comb)

