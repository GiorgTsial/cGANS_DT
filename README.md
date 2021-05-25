# cGANS_DT

The current repository contains code to generate data, run the experiments and recreate results 
for using generative models in order to define digital twins/mirrors of a simulated cantilever 
structure with uncertainties.

Data: the proper datasets can be found in https://drive.google.com/drive/folders/1Ykv_lC1SkRbWtWFoSE_PWK2zd1CgyZqx?usp=sharing and contains the data used in order to recreate the experiments.

BeamSFEM: contains the code needed to run the stochastic finite element simulations.

saved_models: contains the pre-trained models whose results are reported in the paper.

cGAN.py: Module containing the cGAN code.

cGAN_extrapol.py: Module containing the cGAN code required to run extrapolation.

cGAN_extrapolation_testing_results.py: Script to recreate the results of the cGAN extrapolation application.

cGAN_hybrid.py: Module containing the code for the hybrid SFEM-cGAN model.

cGAN_hybrid_extrapol.py: Module containing the code for the hybrid SFEM-cGAN extrapolation application.

cGAN_hybrid_extrapolation_testing_results.py: Script to recreate the results of the hybrid model extrapolation application.

cGAN_linear_testing_results.py: Script to recreate the results of the application of the cGAN on the linear problem.

cGAN_nonlinear_testing_results.py: Script to recreate the results of the application of the cGAN on the nonlinear problem.

fit_SFEM_on_nonlinear.py: Script to find the best parameters for the SFEM model to fit the nonlinear problem data.

hybrid_cGAN_nonlinear_results.py: Script to recreate the results of the application of the hybrid model on the nonlinear problem.

train_cGAN_for_extrapolation.py: Script to train the cGAN according to the reduced dataset, to perform extrapolation on the rest of the data.

train_cGAN_on_linear.py: Script to train the cGAN on the data from the linear problem.

train_cGAN_on_nonlinear.py: Script to train the cGAN on the data from the nonlinear problem.

train_hybrid_cGAN.py: Script to train the hybrid model on the data from the nonlinear problem.

train_hybrid_cGAN_for_extrapolation.py: Script to train the hybrid model on the reduced dataset from the nonlinear problem, to perform extrapolation on the rest of the data.
