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

val_kernels = calculate_validation_data_kernels(validation_data, validation_codes_norm)
noise_dim = 1
sample_dim = 1
code_dim = 1
n_epochs = 10000
n_batch = 512
eval_every_epochs = 100

disc_hidden_sizes = range(20, 2001, 10)
gen_hidden_sizes = range(20, 2001, 10)
repetitions = 10

global_min_kl_divergence = 100000000000000000
best_gen_path = os.path.join(cwd, "best_nonlin_CGAN_generator.h5")
for disc_hidden_size in disc_hidden_sizes:
    gen_hidden_size = disc_hidden_size
    for n in range(repetitions):
        local_min_kl_divergence, best_generator = CGAN_fit(training_data_norm, training_codes_norm, val_kernels,
                                                           noise_dim, sample_dim, code_dim, n_epochs, n_batch,
                                                           eval_every_epochs, disc_hidden_size, gen_hidden_size)
        if local_min_kl_divergence < global_min_kl_divergence:
            global_min_kl_divergence = local_min_kl_divergence
            BESTEST_GENERATOR = best_generator
            BESTEST_GENERATOR.save(best_gen_path)

print(global_min_kl_divergence)
