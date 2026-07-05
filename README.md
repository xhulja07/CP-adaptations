# CP-adaptations
Conformal Prediction Adaptations to Assist the Identification of Correct Predictions

Getting started:
This repository contains the CP adaptations pertaining to the work:
 “On the Use of Conformal Prediction to Assist the Identification of Correct Change-level Fault Predictions”

Repository Structure:
The repository contains 4 folders, named EX<Nr>, corresponding to the four experimental iterations conducted in this work.
Moreover, the repository contains the implementation of the statistical tests (statistical_analysis_CP.py) used to assess the significance of the observed outcomes. 


# Experimental Components:

The conducted experiments leverage existing change-level fault prediction (CL-FP) models, whose implementation has been made publicly available by the corresponding authors.  To train the CL-FP models, we followed the instructions outlined in the corresponding repository.

CL-FP Models:

DeepJIT: https://github.com/soarsmu/DeepJIT/tree/master
LApredict: https://github.com/ZZR0/ISSTA21-JIT-DP
CodeBERT4JIT: https://github.com/Xin-Zhou-smu/Assessing-generalizability-of-CodeBERT

DATASETS:

We conducted our experiments with the QT and OPENSTACK datasets, which can are publicly available under: https://zenodo.org/records/3965246#.XyEDVnUzY5k

CP Models:

To carry out our experiments, we train CP models using the open-source implementation provided by Angelopoulos, Anastasios N & Bates, Stephen: https://github.com/aangelopoulos/conformal-prediction. 
In the first two experimental iterations, we employ the inductive CP algorithm. In the final two iterations, we adopt class-conditional CP.
The implementation of the CP adaptations (CPa, CC-CPa, CPC) can be found inside the implementation folder, inside EX1 (see CPa.py), EX3 (see Class_Conditional_CPa.py), and EX4 (see CPC-py), respectively. Note that EX2, uses the same CPa implementation as EX1, however, this is applied after recalibrating the CL-FP models using Platt scaling and Temperature scaling, respectively.


# Environmental Variables - please install
	python 3.9
	numpy 2.0.2
	pandas 2.3.3
	scipy 1.13.1


# Running the Experiments 

To be able to run the CP adaptations (CPa, CC-CPa, CPC) you need the outputs (prediction probability, prediction label) of the predictions made by the CL-FP models, as well as the true labels of the data instances, as provided in the corresponding dataset. 

How to get them:

- Step 1: Reproduce the results of the CL-FP models under study (DeepJIT/ CodeBERT4JIT/ LApredict)

Get the FP data
Navigate to the corresponding repository of the Fault Dataset that you want to use (QT or OpenStack).
Download the data in your local environment

Get the CL-FP model
Navigate to the corresponding repository of the CL-FP model that you want to work with ( DeepJIT/CodeBERT4JIT/LApredict ).
Clone the repository in your local environment, make sure to install the environment requirements detailed in the repository.
Copy the FP data you downloaded into this repository.
Follow the instructions for running the code (training and evaluation). Compare the accuracy scores with the scores reported in the paper, to ensure that everything runs as it is supposed to do.

- Step 2: Extract the necessary data for running CP and the baseline with the CL-FP models.
Before using the training data (of QT or OpenStack training sets), randomly select 1000 data instances for the calibration set. These instances should then be 'removed' from the training set, i.e., do not use these for training the CL-FP model. 

Training & evaluating the CL-FP model
-Training: Use the rest of the training set to train the CL-FP model, using 10k cross-validation.
-Validation: When validating the CL-FP model: for each instance in the validation set, use the CL-FP model to make a prediction on it, and extract the corresponding prediction probability (Sigmoid score), the predicted label, and the true label (original label on the dataset) of the instance. Store these data in a Excel file, with the following columns: "Predicted prob", "Predicted label", "True label". 
	Example directory path: "{MODEL}/{DATASET}/validation_set" , e.g., "LApredict/QT/validation_set"
-Testing: repeat the same for each instance in the test set. Extract and store the: "Predicted prob", "Predicted label", "True label", make sure to store them in a separate folder form the validation results. 

Example directory path: "{MODEL}/{DATASET}/test_set"
!!! Experiment 1 (EX1) has some sample data files (.xlsx) inside, just to give you an idea of what the input files should look like, to be able to run the experiments.

Constructing the calibration set for the Conformal Predictor
-Training the CP model: After the training of the CL-FP model is complete, use the calibration set on the CL-FP model to make predictions - for each instance in the calibration set, use the trained CL-FP model to make predictions, and extract the: "Predicted prob", "Predicted label", "True label", same as  you did for the validation set. Store these values in an excel file. Make sure to separate them from the validation files. 

Example directory path: "{MODEL}/{DATASET}/calibration_set"

Now you have everything that you need for running the CP adaptations, for example CPa.

- Step 3: run the CPa.py inside the Ex1 directory. 
Set the global variables correctly:
	DATASET = 'qt' or 'os'
	MODEL = 'LApredict' or 'DeepJIT', or 'CodeBERT'
if you want to extend the study with different datasets and CL-FP models, feel free to do so.

Set the CP parameters:
	calib_set_size = 1000  # number of calibration instances for calibrating the CP model
    alpha = 0.05  # (1-alpha) is the desired coverage, in [0, 1].
Feel free to adjust these if you wish, just make sure to understand what they are for. 

By setting "testing = True" inside the _main_ CP will be calibrated on the calibration data and evaluated on the test set data. If "testing = False" , CP will be calibrated on the calibration data and evaluated on the validation set data.

Provide the correct directory paths for the validation_set, test_set, and calibration_set data ("Predicted prob", "Predicted label", "True label") that you extracted above. !!! Note that, when running EX2, these directory paths should point to the folders where you stored the data extracted by the calibrated (via Platt-scaling or other recalibration method) CL-FP model, namely the validation_set, test_set, and calibration_set data ("Calibrated_predicted prob", "Predicted label", "True label").

Run the code. 

The experimental setup of each experiment, can be found in the experimental design section of the original work.