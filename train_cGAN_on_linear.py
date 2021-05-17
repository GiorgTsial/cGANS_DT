from cGAN import *
import os
import json
from keras.models import load_model

# Define the lists of the training, validation and testing keys
training_keys = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 20.0]
validation_keys = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 19.0]
testing_keys = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]

# Load the data
cwd = os.getcwd()
name = "linear_Us.json"
file_path = os.path.join(cwd, "data", name)

with open(file_path, "r") as fp:
    data_dict = json.load(fp)

# Load the data for the cGAN
training_data_norm, training_codes_norm, validation_data, validation_codes_norm, \
testing_data, testing_codes_norm, train_min, train_max = load_real_samples_min_n_max(file_path, training_keys, validation_keys, testing_keys)


val_kernels = calculate_validation_data_kernels(validation_data, validation_codes_norm)
noise_dim = 1
sample_dim = 1
code_dim = 1
n_epochs = 10000
n_batch = 1024
eval_every_epochs = 100

# Define the number of nodes in the hidden layers
disc_hidden_sizes = range(20, 2001, 10)
# gen_hidden_sizes = range(20, 2001, 10)
repetitions = 10

global_min_kl_divergence = 100000000000000000
best_gen_path = os.path.join(cwd, "saved_models", "best_CGAN_generator_linear.h5")
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


