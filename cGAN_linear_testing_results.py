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

# Define the parameters of the model
noise_dim = 1
sample_dim = 1
code_dim = 1

path = os.path.join(cwd, "saved_models",  "CGAN_generator_linear.h5")

model = load_model(path)
model.summary()

testing_avg_kl_diverg = 0
max_kl_divergence = -1.0
for test_code in testing_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=testing_data,
                        testing_codes_norm_value=test_code, validation_codes_norm=testing_codes_norm, train_min=train_min, train_max=train_max)
    testing_avg_kl_diverg += kl_diverg
    if kl_diverg > max_kl_divergence:
        max_kl_divergence = kl_diverg
print("Max KL: ", max_kl_divergence)
testing_avg_kl_diverg /= len(testing_keys)
print("Average KL on testing dataset: ", testing_avg_kl_diverg)

val_avg_kl_diverg = 0
for val_code in validation_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=validation_data,
                                 testing_codes_norm_value=val_code, validation_codes_norm=validation_codes_norm, train_min=train_min, train_max=train_max)
    val_avg_kl_diverg += kl_diverg
val_avg_kl_diverg /= len(validation_keys)

train_kl_diverg = 0
for train_code in training_codes_norm:
    kl_diverg, reals, fakes = test_best_generator_distribs(path, sample_dim=sample_dim, code_dim=code_dim, noise_dim=noise_dim, testing_data=training_data_norm,
                                 testing_codes_norm_value=train_code, validation_codes_norm=training_codes_norm, train_min=train_min, train_max=train_max)
    train_kl_diverg += kl_diverg

print("Mean Testing KL Divegence: ", testing_avg_kl_diverg)
print("Mean Validation KL Divegence: ", val_avg_kl_diverg)
print("Mean Training KL Divegence: ", train_kl_diverg)

