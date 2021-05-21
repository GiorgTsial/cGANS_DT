import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
from keras.activations import linear
from keras.layers import Dense, Lambda
from keras.layers import LeakyReLU, concatenate
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import log
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import os

KD_BANDWIDTH = 0.1


# Define the standalone discriminator model
def define_discriminator(sample_size, code_size, hidden_size=50):
    input_1 = Input(shape=(sample_size, ))
    input_2 = Input(shape=(code_size, ))
    inputs = concatenate([input_1, input_2])
    # Define the discriminator Layers
    d = Dense(hidden_size, kernel_initializer='he_uniform', activation='tanh')(inputs)
    out_classifier = Dense(1, kernel_initializer='he_uniform', activation="sigmoid")(d)
    d_model = Model([input_1, input_2], out_classifier)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))
    return d_model


# define the standalone generator model
def define_generator(data_size, noise_size, code_size, hidden_size=100):
    input_1 = Input(shape=(noise_size, ))
    input_2 = Input(shape=(code_size, ))
    inputs = concatenate([input_1, input_2])
    gen = Dense(hidden_size, kernel_initializer='he_uniform', activation="tanh")(inputs)
    gen_2 = Dense(data_size, kernel_initializer='he_uniform', activation="tanh")(gen)
    code_lin = Lambda(lambda x: x)(input_2)
    model = Model([input_1, input_2], [gen_2, code_lin])
    return model


# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model):
    # make weights in the discriminator (some shared with the q model) as not trainable
    d_model.trainable = False
    # connect g outputs to d inputs
    d_output = d_model(g_model.output)
    # define composite model
    model = Model(g_model.input, d_output)
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy'], optimizer=opt)
    return model


def load_real_samples(file_path, training_keys, validation_keys, testing_keys):
    import json
    from sklearn.decomposition import PCA
    with open(file_path, "r") as fp:
        data_dict = json.load(fp)
    print(data_dict.keys())

    all_keys = [float(i) for i in data_dict.keys()]
    print(all_keys)
    # training_keys = all_keys[0:2] + all_keys[3:-4] + all_keys[-3:]
    # validation_keys = [all_keys[2]]
    # testing_keys = [all_keys[-4]]

    print(training_keys)
    print(validation_keys)
    print(testing_keys)

    training_data = []
    training_codes = []
    for key in training_keys:
        temp_key = str(key)
        training_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            training_codes.append(key)

    training_data = np.hstack(training_data)

    training_codes = np.array(training_codes)
    training_codes = np.reshape(training_codes, (training_codes.shape[0], 1))
    training_data = np.reshape(training_data, (training_data.shape[0], 1))
    training_data = np.concatenate((training_data, training_codes), axis=1)

    validation_data = []
    validation_codes = []
    for key in validation_keys:
        temp_key = str(key)
        validation_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            validation_codes.append(key)

    validation_data = np.hstack(validation_data)

    validation_codes = np.array(validation_codes)
    validation_codes = np.reshape(validation_codes, (validation_codes.shape[0], 1))
    validation_data = np.reshape(validation_data, (validation_data.shape[0], 1))
    validation_data = np.concatenate((validation_data, validation_codes), axis=1)

    testing_data = []
    testing_codes = []
    for key in testing_keys:
        temp_key = str(key)
        testing_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            testing_codes.append(key)

    testing_data = np.hstack(testing_data)

    testing_codes = np.array(testing_codes)
    testing_codes = np.reshape(testing_codes, (testing_codes.shape[0], 1))
    testing_data = np.reshape(testing_data, (testing_data.shape[0], 1))
    testing_data = np.concatenate((testing_data, testing_codes), axis=1)

    print(np.max(training_data, axis=0), np.min(training_data, axis=0))
    # Normalise data
    validation_data = 2 * (validation_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0
    testing_data = 2 * (testing_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0
    training_data_norm = 2 * (training_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0

    training_codes = np.array(training_keys)
    validation_codes = np.array(validation_keys)
    testing_codes = np.array(testing_keys)

    training_codes_norm = 2 * (training_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0
    validation_codes_norm = 2 * (validation_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0
    testing_codes_norm = 2 * (testing_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0

    return training_data_norm, training_codes_norm, validation_data, validation_codes_norm, testing_data, testing_codes_norm


def load_real_samples_min_n_max(file_path, training_keys, validation_keys, testing_keys):
    import json
    from sklearn.decomposition import PCA
    with open(file_path, "r") as fp:
        data_dict = json.load(fp)
    print(data_dict.keys())

    all_keys = [float(i) for i in data_dict.keys()]
    print(all_keys)
    # training_keys = all_keys[0:2] + all_keys[3:-4] + all_keys[-3:]
    # validation_keys = [all_keys[2]]
    # testing_keys = [all_keys[-4]]

    print(training_keys)
    print(validation_keys)
    print(testing_keys)

    training_data = []
    training_codes = []
    for key in training_keys:
        temp_key = str(key)
        training_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            training_codes.append(key)

    training_data = np.hstack(training_data)

    training_codes = np.array(training_codes)
    training_codes = np.reshape(training_codes, (training_codes.shape[0], 1))
    training_data = np.reshape(training_data, (training_data.shape[0], 1))
    training_data = np.concatenate((training_data, training_codes), axis=1)

    validation_data = []
    validation_codes = []
    for key in validation_keys:
        temp_key = str(key)
        validation_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            validation_codes.append(key)

    validation_data = np.hstack(validation_data)

    validation_codes = np.array(validation_codes)
    validation_codes = np.reshape(validation_codes, (validation_codes.shape[0], 1))
    validation_data = np.reshape(validation_data, (validation_data.shape[0], 1))
    validation_data = np.concatenate((validation_data, validation_codes), axis=1)

    testing_data = []
    testing_codes = []
    for key in testing_keys:
        temp_key = str(key)
        testing_data.append(np.array(data_dict[temp_key])[:, -1])
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            testing_codes.append(key)

    testing_data = np.hstack(testing_data)

    testing_codes = np.array(testing_codes)
    testing_codes = np.reshape(testing_codes, (testing_codes.shape[0], 1))
    testing_data = np.reshape(testing_data, (testing_data.shape[0], 1))
    testing_data = np.concatenate((testing_data, testing_codes), axis=1)

    training_data_min = np.min(training_data, axis=0)
    training_data_max = np.max(training_data, axis=0)
    # Normalise data
    validation_data = (2 * (validation_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0) * 0.8
    testing_data = (2 * (testing_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0) * 0.8
    training_data_norm = (2 * (training_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0) * 0.8

    training_codes = np.array(training_keys)
    validation_codes = np.array(validation_keys)
    testing_codes = np.array(testing_keys)

    training_codes_norm = (2 * (training_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0) * 0.8
    validation_codes_norm = (2 * (validation_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0) * 0.8
    testing_codes_norm = (2 * (testing_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0) * 0.8
    return training_data_norm, training_codes_norm, validation_data, validation_codes_norm, testing_data, testing_codes_norm, training_data_min[:-1], training_data_max[:-1]


def load_real_samples_and_pca(file_path, training_keys):
    import json
    from sklearn.decomposition import PCA
    with open(file_path, "r") as fp:
        data_dict = json.load(fp)
    print(data_dict.keys())

    all_keys = [float(i) for i in data_dict.keys()]
    print(all_keys)
    # training_keys = all_keys[0:2] + all_keys[3:-4] + all_keys[-3:]
    validation_keys = [all_keys[2]]
    testing_keys = [all_keys[-4]]

    print(training_keys)
    print(validation_keys)
    print(testing_keys)

    training_data = []
    training_codes = []
    for key in training_keys:
        temp_key = str(key)
        training_data.append(np.array(data_dict[temp_key]))
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            training_codes.append(key)

    training_data = np.vstack(training_data)

    # Set up the pca
    pca = PCA(n_components=3)
    training_data = pca.fit_transform(training_data)

    training_codes = np.array(training_codes)
    training_codes = np.reshape(training_codes, (training_codes.shape[0], 1))
    training_data = np.concatenate((training_data, training_codes), axis=1)

    validation_data = []
    validation_codes = []
    for key in validation_keys:
        temp_key = str(key)
        validation_data.append(np.array(data_dict[temp_key]))
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            validation_codes.append(key)

    validation_data = np.vstack(validation_data)

    validation_data = pca.transform(validation_data)

    validation_codes = np.array(validation_codes)
    validation_codes = np.reshape(validation_codes, (validation_codes.shape[0], 1))
    validation_data = np.concatenate((validation_data, validation_codes), axis=1)

    testing_data = []
    testing_codes = []
    for key in testing_keys:
        temp_key = str(key)
        testing_data.append(np.array(data_dict[temp_key]))
        for i in range(np.array(data_dict[temp_key]).shape[0]):
            testing_codes.append(key)

    testing_data = np.vstack(testing_data)

    testing_data = pca.transform(testing_data)

    testing_codes = np.array(testing_codes)
    testing_codes = np.reshape(testing_codes, (testing_codes.shape[0], 1))
    testing_data = np.concatenate((testing_data, testing_codes), axis=1)

    # Normalise data
    validation_data = 2 * (validation_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0
    testing_data = 2 * (testing_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0
    training_data_norm = 2 * (training_data - np.min(training_data, axis=0)) / (
                np.max(training_data, axis=0) - np.min(training_data, axis=0)) - 1.0

    training_codes = np.array(training_keys)
    validation_codes = np.array(validation_keys)
    testing_codes = np.array(testing_keys)

    training_codes_norm = 2 * (training_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0
    validation_codes_norm = 2 * (validation_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0
    testing_codes_norm = 2 * (testing_codes - np.min(training_codes, axis=0)) / (
                np.max(training_codes, axis=0) - np.min(training_codes, axis=0)) - 1.0

    return training_data_norm, training_codes_norm, validation_data, validation_codes_norm, testing_data, testing_codes_norm, pca


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(noise_dim, code_dim, n_samples, available_codes):
    z_noise = np.random.normal(0, 1, noise_dim * n_samples)
    z_noise = z_noise.reshape(n_samples, noise_dim)
    codes = np.array([np.random.choice(available_codes) for i in range(n_samples)])
    codes = codes.reshape((n_samples, code_dim))
    z_input = np.hstack((z_noise, codes))
    return z_input


def generate_fake_samples(generator, noise_dim, code_dim, n_samples, available_codes):
    z_input = generate_latent_points(noise_dim, code_dim, n_samples, available_codes)
    fake_samples = generator.predict([z_input[:, :noise_dim], z_input[:, noise_dim:]])
    y = np.zeros((n_samples, 1))
    return fake_samples, y


def generate_fake_samples_constant_code(generator, code_dim, noise_dim, code, n_samples, available_codes):
    z_input = generate_latent_points(noise_dim, code_dim, n_samples, available_codes)
    z_input[:, noise_dim:] = np.ones_like(z_input[:, noise_dim:]) * code
    fake_samples = generator.predict([z_input[:, :noise_dim], z_input[:, noise_dim:]])
    return fake_samples[0]


def get_real_data_constant_code(real_data, samples_dim, code):
    code_data = []
    for datum in real_data:
        if datum[samples_dim:] == code:
            code_data.append(datum[:samples_dim])
    return np.array(code_data)


def kl_divergence(p_dist, q_dist, n_samples_per_axis=30, n_axis=2):
    if n_axis == 2:
        x = np.linspace(-1.0, 1.0, n_samples_per_axis)
        y = np.linspace(-1.0, 1.0, n_samples_per_axis)
        grids = np.meshgrid(x, y)
    elif n_axis == 3:
        x = np.linspace(-1.0, 1.0, n_samples_per_axis)
        y = np.linspace(-1.0, 1.0, n_samples_per_axis)
        z = np.linspace(-1.0, 1.0, n_samples_per_axis)
        grids = np.meshgrid(x, y, z)
    elif n_axis == 1:
        grids = np.linspace(-1.0, 1.0, n_samples_per_axis)
    print("Grid complete!")
    if n_axis != 1:
        grid = np.vstack(grids).reshape((n_axis, n_samples_per_axis**n_axis)).T
    else:
        grid = grids
    grid = np.reshape(grid, (grid.shape[0], 1))
    probs_p = np.exp(p_dist.score_samples(grid))
    probs_q = np.exp(q_dist.score_samples(grid))
    print("prob_calc_complete")
    kl = entropy(probs_p, probs_q)
#     for i in range(n_samples_per_axis**n_axis):
#         kl += probs_p[i] * log(probs_p[i] / probs_q[i]) if probs_q[i] != 0 else 0
    return kl


def test_KL_divergence(generator, noise_dim, code_dim, validation_kernels, val_codes, n_samples_per_axis=40, n_axis=1):
    total_KL = 0
    for code in validation_kernels:
        print(code)
        fakes = generate_fake_samples_constant_code(generator, code_dim, noise_dim, code, n_samples_per_axis**n_axis, val_codes)
        print("Fake generation complete")
        fakes_kernel = KernelDensity(kernel='gaussian', bandwidth=KD_BANDWIDTH).fit(fakes)
        print("Kernel complete")
        total_KL += kl_divergence(validation_kernels[code], fakes_kernel, n_samples_per_axis, n_axis)
    return total_KL


def train(generator, discriminator, gan, dataset, available_codes, samples_dim, noise_dim, code_dim, validation_kernels, n_epochs=10000, n_batch=512, evaluate_freq=500, evaluate_after=0):
    half_batch = int(n_batch / 2)
    best_local_g_model = None
    min_local_KL = 100000000000000000
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_real = [X_real[:, :samples_dim], X_real[:, samples_dim:]]

        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(generator, noise_dim, code_dim, half_batch, available_codes)
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
        z_input = generate_latent_points(noise_dim, code_dim, n_batch, available_codes)
        z_input = [z_input[:, :noise_dim], z_input[:, noise_dim:]]
        y_gan = np.ones((n_batch, 1))
        gan_loss = gan.train_on_batch(z_input, y_gan)
        # summarize loss on this batch
        print('>%d, d[%.3f,%.3f], gan_loss[%.3f]' % (i + 1, d_loss_real, d_loss_fake, gan_loss))
        if (i+1) % evaluate_freq == 0 and (i+1) > evaluate_after:
            val_KL_divergence = test_KL_divergence(generator, noise_dim, code_dim, validation_kernels, available_codes, n_axis=samples_dim, n_samples_per_axis=1000)
            print("##############################################################")
            print("#             validation KL Divergence = %.3f                #" % (val_KL_divergence))
            print("##############################################################")
            if val_KL_divergence <= min_local_KL:
                min_local_KL = val_KL_divergence
                best_local_g_model = generator
                cwd = os.getcwd()
                name = "bestest_nonlin_CGAN_generator_extrapol_temp.h5"
                path = os.path.join(cwd, name)
                best_local_g_model.save(path)
    return min_local_KL, best_local_g_model


def calculate_validation_data_kernels(validation_data, validation_codes):
    validation_data_dict = {}
    for code in validation_codes:
        validation_data_dict[code] = []

    for datum in validation_data:
        code = datum[-1]
        validation_data_dict[code].append(datum[:-1])

    for code in validation_data_dict:
        validation_data_dict[code] = np.array(validation_data_dict[code])

    for code in validation_data_dict:
        validation_data_dict[code] = np.array(validation_data_dict[code])
        print(validation_data_dict[code].shape)

    validation_kds = {}
    for code in validation_data_dict:
        kde = KernelDensity(kernel='gaussian', bandwidth=KD_BANDWIDTH).fit(validation_data_dict[code])
        validation_kds[code] = kde

    return validation_kds


def test_best_generator(path, sample_dim, code_dim, noise_dim, testing_data, testing_codes_norm_value, validation_codes_norm):
    best_gen = load_model(path)
    fakes = generate_fake_samples_constant_code(best_gen, code_dim, noise_dim, testing_codes_norm_value, 1000, validation_codes_norm)
    reals = get_real_data_constant_code(testing_data, sample_dim, testing_codes_norm_value)
    plt.hist(fakes, color="r")
    plt.hist(reals, color="b", alpha=0.05)
    plt.xlim([-1.0, 1.0])


def test_best_generator_distribs(path, sample_dim, code_dim, noise_dim, testing_data, testing_codes_norm_value, validation_codes_norm, train_min, train_max):
    best_gen = load_model(path)
    fakes = generate_fake_samples_constant_code(best_gen, code_dim, noise_dim, testing_codes_norm_value, 10000, validation_codes_norm)
    reals = get_real_data_constant_code(testing_data, sample_dim, testing_codes_norm_value)
    kd_real = KernelDensity(bandwidth=KD_BANDWIDTH)
    kd_fakes = KernelDensity(bandwidth=KD_BANDWIDTH)
    kd_real.fit(reals)
    kd_fakes.fit(fakes)

    x = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    real_scores = np.exp(kd_real.score_samples(x))

    fake_scores = np.exp(kd_fakes.score_samples(x))

    kl_diverg = entropy(real_scores, fake_scores)

    x_new = (x + 0.8) / 1.6 * (train_max - train_min) + train_min

    plt.plot(x_new, fake_scores, color="r")
    plt.plot(x_new, real_scores, color="b")
    plt.xlabel("Tip displacement", fontsize=14)
    plt.ylabel("Probability density", fontsize=14)
    plt.xlim([0.0, 0.07])
    plt.ylim([0.0, 4.5])
    plt.show()
    return kl_diverg, reals, fakes


def CGAN_fit(training_data_norm, training_codes_norm, val_kernels, noise_dim, sample_dim, code_dim, n_epochs, n_batch, eval_every_epochs, disc_hidden_size, gen_hidden_size):
    discriminator = define_discriminator(sample_dim, code_dim, disc_hidden_size)
    generator = define_generator(sample_dim, noise_dim, code_dim, gen_hidden_size)
    gan = define_gan(generator, discriminator)
    min_kl_divergence, best_generator = train(generator, discriminator, gan, training_data_norm, training_codes_norm, sample_dim, noise_dim, code_dim,
          val_kernels, n_epochs, n_batch, eval_every_epochs)
    return min_kl_divergence, best_generator
