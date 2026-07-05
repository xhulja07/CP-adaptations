import numpy as np
import pandas as pd
import os
import sys

DATASET = 'os'
# DATASET variable takes values 'os' or 'qt'
MODEL = 'LApredict'
# CL-FP MODELs: DeepJIT, CodeBERT4JIT, LApredict

predictions_length = []
CP_validity = []
size_0_sets = []
size_1_sets = []
size_2_sets = []
size_1_sets_correct = []
size_1_sets_incorrect = []
size_1_sets_correct_class1 = []
size_1_sets_correct_class0 = []
size_1_sets_incorrect_class1 = []
size_1_sets_incorrect_class0 = []

predictions_length_a1 = []
CP_validity_a1 = []
size_0_sets_a1 = []
size_1_sets_a1 = []
size_2_sets_a1 = []
size_1_sets_correct_a1 = []
size_1_sets_incorrect_a1 = []
size_1_sets_correct_class1_a1 = []
size_1_sets_correct_class0_a1 = []
size_1_sets_incorrect_class1_a1 = []
size_1_sets_incorrect_class0_a1 = []

predictions_length_a15 = []
CP_validity_a15 = []
size_0_sets_a15 = []
size_1_sets_a15 = []
size_2_sets_a15 = []
size_1_sets_correct_a15 = []
size_1_sets_incorrect_a15 = []
size_1_sets_correct_class1_a15 = []
size_1_sets_correct_class0_a15 = []
size_1_sets_incorrect_class1_a15 = []
size_1_sets_incorrect_class0_a15 = []


def get_predictions_before_after_calibration(file_path):
    df = pd.read_excel(file_path)
    # Extract each column into separate lists (excluding the 'index' column)
    input_list = df['Input'].tolist()
    true_label_list = df['True label'].tolist()
    predicted_prob_uncalibrated_list = df['Predicted prob before calibration'].tolist()
    predicted_prob_calibrated_list = df['Predicted prob after calibration'].tolist()
    return predicted_prob_uncalibrated_list, predicted_prob_calibrated_list, true_label_list, len(true_label_list)

def get_predictions(file_path):
    df = pd.read_excel(file_path)
    # Extract each column into separate lists (excluding the 'index' column)
    #input_list = df['Input'].tolist()
    true_label_list = df['True label'].tolist()
    predicted_prob = df['Predicted prob'].tolist()
    return predicted_prob, true_label_list, len(true_label_list)


def extract_statistics(set_length, cp_validity, size0, size1, size2, size1Corr, size1Inc, size1Corr_c1, size1Corr_c0, size1Inc_c1, size1Inc_c0, alpha= 0.05):
    if alpha == 0.05:
        predictions_length.append(set_length)
        CP_validity.append(cp_validity)
        size_0_sets.append(size0)
        size_1_sets.append(size1)
        size_2_sets.append(size2)
        size_1_sets_correct.append(size1Corr)
        size_1_sets_incorrect.append(size1Inc)
        size_1_sets_correct_class1.append(size1Corr_c1)
        size_1_sets_correct_class0.append(size1Corr_c0)
        size_1_sets_incorrect_class1.append(size1Inc_c1)
        size_1_sets_incorrect_class0.append(size1Inc_c0)
    elif alpha == 0.10:
        predictions_length_a1.append(set_length)
        CP_validity_a1.append(cp_validity)
        size_0_sets_a1.append(size0)
        size_1_sets_a1.append(size1)
        size_2_sets_a1.append(size2)
        size_1_sets_correct_a1.append(size1Corr)
        size_1_sets_incorrect_a1.append(size1Inc)
        size_1_sets_correct_class1_a1.append(size1Corr_c1)
        size_1_sets_correct_class0_a1.append(size1Corr_c0)
        size_1_sets_incorrect_class1_a1.append(size1Inc_c1)
        size_1_sets_incorrect_class0_a1.append(size1Inc_c0)
    elif alpha == 0.15:
        predictions_length_a15.append(set_length)
        CP_validity_a15.append(cp_validity)
        size_0_sets_a15.append(size0)
        size_1_sets_a15.append(size1)
        size_2_sets_a15.append(size2)
        size_1_sets_correct_a15.append(size1Corr)
        size_1_sets_incorrect_a15.append(size1Inc)
        size_1_sets_correct_class1_a15.append(size1Corr_c1)
        size_1_sets_correct_class0_a15.append(size1Corr_c0)
        size_1_sets_incorrect_class1_a15.append(size1Inc_c1)
        size_1_sets_incorrect_class0_a15.append(size1Inc_c0)


def write_statistics(filename, filename1, filename2):

    df = pd.DataFrame({
        'nr_instances': predictions_length,
        'CP_validity': CP_validity,
        'size_0_sets': size_0_sets,
        'size_1_sets': size_1_sets,
        'size_2_sets': size_2_sets,
        'size_1_sets_correct': size_1_sets_correct,
        'size_1_sets_incorrect': size_1_sets_incorrect,
        'size_1_sets_correct_class1': size_1_sets_correct_class1,
        'size_1_sets_correct_class0': size_1_sets_correct_class0,
        'size_1_sets_incorrect_class1': size_1_sets_incorrect_class1,
        'size_1_sets_incorrect_class0': size_1_sets_incorrect_class0
    })
    df.to_excel(filename, index=False)

    df1 = pd.DataFrame({
        'nr_instances': predictions_length_a1,
        'CP_validity': CP_validity_a1,
        'size_0_sets': size_0_sets_a1,
        'size_1_sets': size_1_sets_a1,
        'size_2_sets': size_2_sets_a1,
        'size_1_sets_correct': size_1_sets_correct_a1,
        'size_1_sets_incorrect': size_1_sets_incorrect_a1,
        'size_1_sets_correct_class1': size_1_sets_correct_class1_a1,
        'size_1_sets_correct_class0': size_1_sets_correct_class0_a1,
        'size_1_sets_incorrect_class1': size_1_sets_incorrect_class1_a1,
        'size_1_sets_incorrect_class0': size_1_sets_incorrect_class0_a1
    })
    df1.to_excel(filename1, index=False)

    df2 = pd.DataFrame({
        'nr_instances': predictions_length_a15,
        'CP_validity': CP_validity_a15,
        'size_0_sets': size_0_sets_a15,
        'size_1_sets': size_1_sets_a15,
        'size_2_sets': size_2_sets_a15,
        'size_1_sets_correct': size_1_sets_correct_a15,
        'size_1_sets_incorrect': size_1_sets_incorrect_a15,
        'size_1_sets_correct_class1': size_1_sets_correct_class1_a15,
        'size_1_sets_correct_class0': size_1_sets_correct_class0_a15,
        'size_1_sets_incorrect_class1': size_1_sets_incorrect_class1_a15,
        'size_1_sets_incorrect_class0': size_1_sets_incorrect_class0_a15
    })
    df2.to_excel(filename2, index=False)

def clear_all_lists():
    predictions_length.clear()
    CP_validity.clear()
    size_0_sets.clear()
    size_1_sets.clear()
    size_2_sets.clear()
    size_1_sets_correct.clear()
    size_1_sets_incorrect.clear()
    size_1_sets_correct_class1.clear()
    size_1_sets_correct_class0.clear()
    size_1_sets_incorrect_class1.clear()
    size_1_sets_incorrect_class0.clear()

    predictions_length_a1.clear()
    CP_validity_a1.clear()
    size_0_sets_a1.clear()
    size_1_sets_a1.clear()
    size_2_sets_a1.clear()
    size_1_sets_correct_a1.clear()
    size_1_sets_incorrect_a1.clear()
    size_1_sets_correct_class1_a1.clear()
    size_1_sets_correct_class0_a1.clear()
    size_1_sets_incorrect_class1_a1.clear()
    size_1_sets_incorrect_class0_a1.clear()

    predictions_length_a15.clear()
    CP_validity_a15.clear()
    size_0_sets_a15.clear()
    size_1_sets_a15.clear()
    size_2_sets_a15.clear()
    size_1_sets_correct_a15.clear()
    size_1_sets_incorrect_a15.clear()
    size_1_sets_correct_class1_a15.clear()
    size_1_sets_correct_class0_a15.clear()
    size_1_sets_incorrect_class1_a15.clear()
    size_1_sets_incorrect_class0_a15.clear()



def compute_q_hat(predictions, labels, alpha, n):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    x = float(1)
    class_1 = np.array(predictions)
    class_0 = x-class_1
    true_class_prob = []
    conformal_scores = []

    print("Alpha value:", alpha)
    for i in range(len(predictions)):
        if labels[i] == 0:
            pred_score = class_0[i]
        else:
            pred_score = class_1[i]

        true_class_prob.append(pred_score)
        # compute conformal score: si = 1 − ˆf (Xi)Yi
        conformal_scores.append(float(x-pred_score))
    #compute quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(conformal_scores, q_level, method='higher')
    return qhat


def CPa(t_predictions, t_true_labels, quantile, alpha, calib='uncalibrated', index=0):
    # Make sure t_predictions is a list or array, otherwise convert it
    if isinstance(t_predictions, pd.Series) or isinstance(t_predictions, pd.DataFrame):
        t_predictions = t_predictions.tolist()

    res = []
    for t in t_predictions:
        t_prob = get_classes_prob(t)
        set, size = compute_conf_prediction_set(quantile, t_prob)
        #get the true label
        tl = t_true_labels[t_predictions.index(t)]
        correct = -1
        correct_c1 = -1
        incorrect_c1= -1
        # check the size of the prediction set computed by CP
        if size == 0:
            correct = -2
        elif size == 1:
            #check if the prediction set contains the true label
            if set[0][1] == tl:
                correct = 1
                # check if the true label is class C1 or C0
                if tl == 1:
                    correct_c1 = 1
                else:
                    correct_c1 = 0
            else:
                correct = 0
                if tl == 1:
                    incorrect_c1 = 1
                else:
                    incorrect_c1 = 0

        res.append((t_prob, size, set, tl, correct, correct_c1, incorrect_c1))

    df = pd.DataFrame(res, columns=['softmax', 'size', 'conf set', 'true label', 'Conf pred check', 'correct_size1_class', 'incorrect_size1_class'])
    df.to_excel(f'{MODEL}/{DATASET}/testing-ALPHA' + str(alpha) + '_calibration_' + str(calib) + '_res_' + str(DATASET) + '_' + str(index) + '.xlsx')
    cp_validity = (float) (len(df[df['Conf pred check'] == 1 ]) + len(df[df['Conf pred check'] == -1 ]))/ len(t_predictions)
    extract_statistics(len(t_predictions), cp_validity, len(df[df['size'] == 0]), len(df[df['size'] == 1]), len(df[df['size'] == 2]), len(df[df['Conf pred check'] == 1]), len(df[df['Conf pred check'] == 0 ]), len(df[df['correct_size1_class'] == 1]), len(df[df['correct_size1_class'] == 0]), len(df[df['incorrect_size1_class'] == 1]),  len(df[df['incorrect_size1_class'] == 0]), alpha)

def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_1, class_0])

def compute_conf_prediction_set(qhat, predict_scores):
    prediction_set = []
    if predict_scores[0] >= (1-qhat):
        prediction_set.append((predict_scores[0], 1))
    if predict_scores[1] >= (1-qhat):
        prediction_set.append((predict_scores[1], 0))
    return prediction_set, len(prediction_set)



if __name__ == '__main__':

    # update calib size to the nr of rows in the file
    #calib_set_size = 1000  # number of calibration instances
    alpha_0 = 0.05  # 1-alpha is the desired coverage
    alpha_1 = 0.10  # 1-alpha is the desired coverage
    alpha_2 = 0.15  # 1-alpha is the desired coverage
    testing = True  # 'testing' controls whether the predictions will be made on the validation or on the test set: True=> on the test set; False=> on the valid set

    if testing is True:
        save_file = f'{MODEL}/{DATASET}/CP_with_{DATASET}_test_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename05 = f'{MODEL}/CP_with_{DATASET}_alpha_0.05_test_set.xlsx'
        filename1 = f'{MODEL}/CP_with_{DATASET}_alpha_0.1_test_set.xlsx'
        filename15 = f'{MODEL}/CP_with_{DATASET}_alpha_0.15_test_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/test_set'
    else:
        save_file = f'CP_res/{MODEL}/{DATASET}/CP_with_{DATASET}_validation_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename05 = f'{MODEL}/CP_with_{DATASET}_alpha_0.05_validation_set.xlsx'
        filename1 = f'{MODEL}/CP_with_{DATASET}_alpha_0.1_validation_set.xlsx'
        filename15 = f'{MODEL}/CP_with_{DATASET}_alpha_0.15_validation_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/validation_set'


    filesInCalib = sorted(os.listdir(folderCalib))
    filesInEval = sorted(os.listdir(folderEval))
    print("Calibration folder", folderCalib)
    print("Test folder", folderEval)
    num_files = len(filesInCalib)
    print("Nr files inside Calib folder:", num_files)
    #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
    for i in range(num_files):
        print("Run uncalibrated nr:", i)
        file_calib = filesInCalib[i]
        file_eval = filesInEval[i]
        # step 1: use the trained model to make predictions on the calibration set
        print(f"Calibrating the model on Calib data - file {file_calib}")

        predictions, true_labels, calib_set_size = get_predictions(folderCalib + '/' + file_calib)

        # step 2: compute the empirical quantile (q_hat) using the calibration set (predictions and true labels)
        quantile_a0 = compute_q_hat(predictions, true_labels, alpha_0, calib_set_size)
        quantile_a1 = compute_q_hat(predictions, true_labels, alpha_1, calib_set_size)
        quantile_a2 = compute_q_hat(predictions, true_labels, alpha_2, calib_set_size)
        # step 3: use the trained model to make predictions on the test data, or the validation data
        if testing is True:
            print(f"Evaluating the CP model on Test data - file {file_eval}")
        else:
            print(f"Evaluating the CP model on Valid data - file {file_eval}")

        t_predictions, t_true_labels, test_set_size = get_predictions(folderEval + '/' + file_eval)

        # step 4: apply conformal prediction (CPa) on the validation/test set predictions
        CPa(t_predictions, t_true_labels, quantile_a0, alpha_0, index=i)
        CPa(t_predictions, t_true_labels, quantile_a1, alpha_1, index=i)
        CPa(t_predictions, t_true_labels, quantile_a2, alpha_2, index=i)

        # save data
        file = open(save_file, "a")  # see main.py
        file.write(" {} dataset experiment nr: {}: ".format(DATASET, i) + "\n")
        file.write(" Conformal set size: " + str(calib_set_size) + "\n")
        file.write(" Evaluation set size: " + str(test_set_size) + "\n")
        file.write(
            " qhat_0: {0:f}:, qhat_1: {1:f}:, qhat_2: {2:f}: ".format(quantile_a0, quantile_a1, quantile_a2) + "\n")
        file.write(" Alpha_0: {0:f}, Alpha_1: {1:f}, Alpha_2: {2:f} c: ".format(alpha_0, alpha_1, alpha_2) + "\n")
        file.write("------------------------------------------------------------ \n")
        file.close()

    write_statistics(filename05, filename1, filename15)
    clear_all_lists()
