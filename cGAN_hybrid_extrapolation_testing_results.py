from BeamSFEM.Beam_PlainStress import Beam as beam_linear
from cGAN_hybrid_extrapol import *
import os
import json
import numpy as np

training_keys = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0]
validation_keys = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0]
testing_keys = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]


training_keys = training_keys + validation_keys + testing_keys

# Define SFEM parameters
base_E = float(2.50000000e+09)
E_stdv = 2.57894737e-01
h = 0.4
L = 5.0
rho = 7800
T = 30.0
t = 0.1
CorrLength = [3.05000000e+00]
base_element_size = 0.1
zeta = 0.05
poisson = 0.3

run_SFEM = False
load_SFEM_noise = True

if run_SFEM:
    beam = beam_linear(base_E=base_E, E_stdv=E_stdv, h=h, L=L, rho=rho, T=T, t=t,
                                    base_element_size=base_element_size, zeta=zeta, poisson=poisson, CorrLength=CorrLength)

    all_SFEM_data = []

    training_noise_dict = {}

    for load in training_keys:
        Us, CoVU = beam.StaticSSFEM(load=load)
        SFEM_data = beam.generate_static_samples_SSFEM(10000, Us=Us)[:, -1]
        all_SFEM_data.append(SFEM_data)
        training_noise_dict[load] = SFEM_data

    all_SFEM_data = np.hstack(all_SFEM_data)
    min_val = np.min(all_SFEM_data, axis=0)
    max_val = np.max(all_SFEM_data, axis=0)

    training_keys_ar = np.array(training_keys)

    training_keys_norm = 2 * (training_keys_ar - np.min(training_keys_ar, axis=0)) / (np.max(training_keys_ar, axis=0) - np.min(training_keys_ar, axis=0)) - 1.0

    to_save_dict = {}
    for i, load in enumerate(training_keys):
        training_noise_dict[load] = 2 * (training_noise_dict[load] - min_val) / (max_val - min_val) - 1.0
        print(training_noise_dict[load].shape)
        to_save_dict[str(training_keys_norm[i])] = training_noise_dict[load].tolist()

    cwd = os.getcwd()
    dict_path = os.path.join(cwd, "data", "training_noise_dict.json")
    with open(dict_path, "w") as fp:
        json.dump(to_save_dict, fp)

if load_SFEM_noise:
    cwd = os.getcwd()
    dict_path = os.path.join(cwd, "data", "training_noise_dict.json")
    with open(dict_path, "r") as fp:
        to_save_dict = json.load(fp)
    training_noise_dict = {}
    for key in to_save_dict.keys():
        training_noise_dict[float(key)] = np.array(to_save_dict[key])
    print(training_noise_dict[1.0])
    print(training_noise_dict.keys())


training_keys = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0]
validation_keys = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0]
testing_keys = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
                39.0, 40.0]

testing_keys_normal = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
testing_keys_extrapol = [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]

cwd = os.getcwd()
name = "nonlin_Us.json"
file_path = os.path.join(cwd, "data", name)

training_data_norm, training_codes_norm, validation_data, validation_codes_norm, \
testing_data, testing_codes_norm, train_min, train_max = load_real_samples_min_n_max(file_path, training_keys, validation_keys, testing_keys)

print(validation_data[:, -1])
print(validation_codes_norm)


validation_codes_norm_new = []
for datum in validation_data:
    code = datum[-1]
    validation_codes_norm_new.append(code)
validation_codes_norm_new = list(set(validation_codes_norm_new))

validation_codes_norm = validation_codes_norm_new

all_keys = np.hstack((training_codes_norm, validation_codes_norm, testing_codes_norm)).tolist()
all_keys = sorted(all_keys)

new_training_noise_dict = {}
for i, key in enumerate(sorted(training_noise_dict.keys())):
    new_training_noise_dict[all_keys[i]] = training_noise_dict[key]

training_noise_dict = new_training_noise_dict

val_kernels = calculate_validation_data_kernels(validation_data, validation_codes_norm)
noise_dim = 1
sample_dim = 1
code_dim = 1


cwd = os.getcwd()
name = "nonlin_Us.json"
file_path = os.path.join(cwd, "data", name)

testing_avg_kl_diverg = 0
max_KL = -1.0

path = os.path.join(cwd, "saved_models", "CGAN_hybrid_generator_extrapol_results.h5")

model = load_model(path)
model.summary()

for i, test_code in enumerate(testing_codes_norm):
    if testing_keys[i] in testing_keys_normal:
        print(testing_keys[i])
        kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=testing_data,
                            testing_codes_norm_value=test_code, validation_codes_norm=testing_codes_norm, train_min=train_min, train_max=train_max, training_noise_dict=training_noise_dict)
        testing_avg_kl_diverg += kl_diverg
testing_avg_kl_diverg /= len(testing_keys_normal)
print("Mean Testing KL Divegence: ", testing_avg_kl_diverg)

testing_avg_kl_diverg_extrapol = 0
for i, test_code in enumerate(testing_codes_norm):
    if testing_keys[i] in testing_keys_extrapol:
        print(testing_keys[i], test_code)
        kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=testing_data,
                            testing_codes_norm_value=test_code, validation_codes_norm=testing_codes_norm, train_min=train_min, train_max=train_max, training_noise_dict=training_noise_dict)
        testing_avg_kl_diverg_extrapol += kl_diverg
testing_avg_kl_diverg_extrapol /= len(testing_keys_extrapol)
print("Mean Testing KL Divegence extrapolation: ", testing_avg_kl_diverg_extrapol)

val_avg_kl_diverg = 0
for val_code in validation_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=validation_data,
                                 testing_codes_norm_value=val_code, validation_codes_norm=validation_codes_norm, train_min=train_min, train_max=train_max, training_noise_dict=training_noise_dict)
    val_avg_kl_diverg += kl_diverg
val_avg_kl_diverg /= len(validation_keys)

print("Mean Validation KL Divegence: ", val_avg_kl_diverg)

for train_code in training_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=training_data_norm,
                                 testing_codes_norm_value=train_code, validation_codes_norm=training_codes_norm, train_min=train_min, train_max=train_max, training_noise_dict=training_noise_dict)




