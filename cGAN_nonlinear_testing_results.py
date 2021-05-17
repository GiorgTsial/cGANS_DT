from cGAN import *
import os
from matplotlib import pyplot as plt

training_keys = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0]
validation_keys = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0]
testing_keys = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0]

cwd = os.getcwd()
name = "nonlin_Us.json"
file_path = os.path.join(cwd, "data", name)

training_data_norm, training_codes_norm, validation_data, validation_codes_norm, \
testing_data, testing_codes_norm, train_min, train_max = load_real_samples_min_n_max(file_path, training_keys, validation_keys, testing_keys)

loadcases = training_data_norm[:, 1]
samples = training_data_norm[:, 0]

loadcases = (loadcases + 1.0) / 2 * (400.0 - 10.0) + 10.0
samples = (samples + 1.0) / 2 * (train_max - train_min) + train_min

plt.scatter(loadcases, samples, c="b")
plt.xlabel("Load", fontsize=14)
plt.ylabel("Tip displacement", fontsize=14)
plt.show()


noise_dim = 10
sample_dim = 1
code_dim = 1

path = os.path.join(cwd, "saved_models", "CGAN_generator_nonlin.h5")

model = load_model(path)
model.summary()

testing_avg_kl_diverg = 0
max_KL = -1.0
for test_code in testing_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=testing_data,
                        testing_codes_norm_value=test_code, validation_codes_norm=testing_codes_norm, train_min=train_min, train_max=train_max)
    testing_avg_kl_diverg += kl_diverg
    if kl_diverg > max_KL:
        max_KL = kl_diverg

print("Max KL: ", max_KL)
testing_avg_kl_diverg /= len(testing_keys)

val_avg_kl_diverg = 0
for val_code in validation_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=validation_data,
                                 testing_codes_norm_value=val_code, validation_codes_norm=validation_codes_norm, train_min=train_min, train_max=train_max)
    val_avg_kl_diverg += kl_diverg
val_avg_kl_diverg /= len(validation_keys)

for train_code in training_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=training_data_norm,
                                 testing_codes_norm_value=train_code, validation_codes_norm=training_codes_norm, train_min=train_min, train_max=train_max)

print("Mean Testing KL Divegence: ", testing_avg_kl_diverg)
print("Mean Validation KL Divegence: ", val_avg_kl_diverg)