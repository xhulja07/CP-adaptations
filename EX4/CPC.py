import numpy as np
import pandas as pd
import os
import sys

DATASET = 'os'
# DATASET variable takes values 'os' or 'qt'
MODEL = 'LApredict'
# CL-FP MODELs: DeepJIT, CodeBERT4JIT, LApredict

c_threshold=0.5
#c_threshold is different for each CL-FP model, based on the optimal threshold identified in EX3 (for achieving TPR=50%)

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

predictions_length_cc = []
defective_labels = []
CC0_nr_sets1 = []
CC1_nr_sets1 = []
CC0_validity_c1 = []
CC0_validity_c2 = []
CC0_validity_c3 = []
CC0_validity_c4 = []
CC1_validity_c1 = []
CC1_validity_c2 = []
CC1_validity_c3 = []
CC1_validity_c4 = []
CC0_correct = []
CC1_correct = []
CC_CPC_C1_c0 = []
CC_CPC_C2_c0 = []
CC_CPC_C3_c0 = []
CC_CPC_C4_c0 = []
CC_CPC_C1_c1 = []
CC_CPC_C2_c1 = []
CC_CPC_C3_c1 = []
CC_CPC_C4_c1 = []

CPC_predictions_length = []
CPC_C1_validity = []
CPC_C2_validity = []
CPC_C3_validity = []
CPC_C4_validity = []
CPC_size_0_sets = []
CPC_size_1_sets = []
CPC_size_2_sets = []
CPC_C1_correct = []
CPC_C2_correct = []
CPC_C3_correct = []
CPC_C4_correct = []
CPC_C1 = []
CPC_C2 = []
CPC_C3 = []
CPC_C4 = []
CPC_c1_cat1 =[]
CPC_c1_cat2 =[]
CPC_c1_cat3 =[]
CPC_c1_cat4 =[]



def get_predictions(file_path):
    df = pd.read_excel(file_path)
    # Extract each column into separate lists (excluding the 'index' column)
    #input_list = df['Input'].tolist()
    true_label_list = df['True label'].tolist()
    predicted_prob = df['Predicted prob'].tolist()
    return predicted_prob, true_label_list, len(true_label_list)

def read_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    uncalib_probabilities = df.iloc[:,1].tolist()
    labels = df.iloc[:,3].tolist()
    platt_probabilities = df.iloc[:,5].tolist()
    return uncalib_probabilities, platt_probabilities, labels, len(labels)

"""
def extract_cc_statistics(length, cc0_validity_c1, cc0_validity_c2, cc0_validity_c3, cc0_validity_c4, cc1_validity_c1, cc1_validity_c2, cc1_validity_c3, cc1_validity_c4, correct_c0, correct_c1, C1_c0, C2_c0, C3_c0, C4_c0, C1_c1, C2_c1, C3_c1, C4_c1):
    predictions_length_cc.append(length)
    #EX4: extract the statistics of each certainty class
    CC0_validity_c1.append(cc0_validity_c1)
    CC0_validity_c2.append(cc0_validity_c2)
    CC0_validity_c3.append(cc0_validity_c3)
    CC0_validity_c4.append(cc0_validity_c4)
    CC1_validity_c1.append(cc1_validity_c1)
    CC1_validity_c2.append(cc1_validity_c2)
    CC1_validity_c3.append(cc1_validity_c3)
    CC1_validity_c4.append(cc1_validity_c4)
    CC0_correct.append(correct_c0)
    CC1_correct.append(correct_c1)
    CC-CPC_C1_c0.append(C1_c0)
    CC-CPC_C2_c0.append(C2_c0)
    CC-CPC_C3_c0.append(C3_c0)
    CC-CPC_C4_c0.append(C4_c0)
    CC-CPC_C1_c1.append(C1_c1)
    CC-CPC_C2_c1.append(C2_c1)
    CC-CPC_C3_c1.append(C3_c1)
    CC-CPC_C4_c1.append(C4_c1)
"""
def extract_cc_statistics(length, nr_labels_c1, cc0_validity_c1, nr_set_1_c0, correct_c0, cc1_validity_c1, nr_set_1_c1 , correct_c1):
    predictions_length_cc.append(length)
    defective_labels.append(nr_labels_c1)
    CC0_validity_c1.append(cc0_validity_c1)
    CC0_nr_sets1.append(nr_set_1_c0)
    CC0_correct.append(correct_c0)
    CC1_validity_c1.append(cc1_validity_c1)
    CC1_nr_sets1.append(nr_set_1_c1)
    CC1_correct.append(correct_c1)

def extract_statistics(set_length, cp_validity, size0, size1, size2, size1Corr, size1Inc, size1Corr_c1, size1Corr_c0, size1Inc_c1, size1Inc_c0, alpha= 0.05):

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

def extract_CPC_statistics(len_predictions, C1_validity, C2_validity, C3_validity, C4_validity,  len_size_0, len_size_1, len_size_2, correct_c1, correct_c2, correct_c3, correct_c4 , c1, c2, c3, c4, nr_label1_c1, nr_label1_c2, nr_label1_c3, nr_label1_c4):
    CPC_predictions_length.append(len_predictions)
    CPC_C1_validity.append(C1_validity)
    CPC_C2_validity.append(C2_validity)
    CPC_C3_validity.append(C3_validity)
    CPC_C4_validity.append(C4_validity)
    CPC_size_0_sets.append(len_size_0)
    CPC_size_1_sets.append(len_size_1)
    CPC_size_2_sets.append(len_size_2)
    CPC_C1_correct.append(correct_c1)
    CPC_C2_correct.append(correct_c2)
    CPC_C3_correct.append(correct_c3)
    CPC_C4_correct.append(correct_c4)
    CPC_C1.append(c1)
    CPC_C2.append(c2)
    CPC_C3.append(c3)
    CPC_C4.append(c4)
    CPC_c1_cat1.append(nr_label1_c1)
    CPC_c1_cat2.append(nr_label1_c2)
    CPC_c1_cat3.append(nr_label1_c3)
    CPC_c1_cat4.append(nr_label1_c4)



def write_cc_statistics(filename):

    df1 = pd.DataFrame({
        'nr_instances': predictions_length_cc,
        'Class=0_validity_c1': CC0_validity_c1,
        'Class=0_validity_c2': CC0_validity_c2,
        'Class=0_validity_c3': CC0_validity_c3,
        'Class=0_validity_c4': CC0_validity_c4,
        'Class=0_correct': CC0_correct,
        'Class=0_cat=1': CC_CPC_C1_c0,
        'Class=0_cat=2': CC_CPC_C2_c0,
        'Class=0_cat=3': CC_CPC_C3_c0,
        'Class=0_cat=4': CC_CPC_C4_c0,
        'Class=1_validity_c1': CC1_validity_c1,
        'Class=1_validity_c2': CC1_validity_c2,
        'Class=1_validity_c3': CC1_validity_c3,
        'Class=1_validity_c4': CC1_validity_c4,
        'Class=1_correct': CC1_correct,
        'Class=1_cat=1': CC_CPC_C1_c1,
        'Class=1_cat=2': CC_CPC_C2_c1,
        'Class=1_cat=3': CC_CPC_C3_c1,
        'Class=1_cat=4': CC_CPC_C4_c1
    })
    df1.to_excel(filename, index=False)

def write_CPC_statistis(filename):

    df1 = pd.DataFrame({
        'nr_instances': CPC_predictions_length,
        'size_1_sets_alpha1': CPC_size_0_sets,
        'size_1_sets_alpha2': CPC_size_1_sets,
        'size_1_sets_alpha3': CPC_size_2_sets,
        'C1_validity': CPC_C1_validity,
        'C2_validity': CPC_C2_validity,
        'C3_validity': CPC_C3_validity,
        'C4_validity': CPC_C4_validity,
        'C1_size': CPC_C1,
        'C2_size': CPC_C2,
        'C3_size': CPC_C3,
        'C4_size': CPC_C4,
        'Correct_C1': CPC_C1_correct,
        'Correct_C2': CPC_C2_correct,
        'Correct_C3': CPC_C3_correct,
        'Correct_C4': CPC_C4_correct,
        'Nr_(C=1)_defective_in_C1': CPC_c1_cat1,
        'Nr_defective_in_C2': CPC_c1_cat2,
        'Nr_defective_in_C3': CPC_c1_cat3,
        'Nr_defective_in_C4': CPC_c1_cat4
    })
    df1.to_excel(filename, index=False)

def write_statistics(filename, filename1):

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
        'size_1_sets_incorrect_class0': size_1_sets_incorrect_class0_a1,
        'nr_instances': predictions_length_cc,
        'nr_defective_instances': defective_labels,
        'CP_validity_class0': CC0_validity_c1,
        'size_1_sets_class0': CC0_nr_sets1,
        'nr_correct_size1_sets_c0': CC0_correct,
        'CP_validity_class1': CC1_validity_c1,
        'size_1_sets_class1': CC1_nr_sets1,
        'nr_correct_size1_sets_c1': CC1_correct
    })
    df1.to_excel(filename, index=False)

    df = pd.DataFrame({
        'nr_instances': predictions_length_cc,
        'nr_defective_instances': defective_labels,
        'CP_validity_class0': CC0_validity_c1,
        'size_1_sets_class0': CC0_nr_sets1,
        'nr_correct_size1_sets_c0': CC0_correct,
        'CP_validity_class1': CC1_validity_c1,
        'size_1_sets_class1': CC1_nr_sets1,
        'nr_correct_size1_sets_c1': CC1_correct
    })
    #df.to_excel(filename1, index=False)


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

    predictions_length_cc.clear()

    CPC_predictions_length.clear()
    CPC_C1_validity.clear()
    CPC_C2_validity.clear()
    CPC_C3_validity.clear()
    CPC_C4_validity.clear()
    CPC_size_0_sets.clear()
    CPC_size_1_sets.clear()
    CPC_size_2_sets.clear()
    CPC_C1.clear()
    CPC_C2.clear()
    CPC_C3.clear()
    CPC_C4.clear()

def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_0, class_1])


def get_predicted_label(predict_scores, c_threshold):
    if (predict_scores >= c_threshold):
        return 1
    else:
        return 0

def get_confusion_matrix_label(predicted_label, true_label):

    if predicted_label == true_label:
        if true_label == 0:
            cm = 'TN'
        else:
            cm = 'TP'
    else:
        if predicted_label == 0:
            cm = 'FN'
        else:
            cm = 'FP'
    return cm


def compute_conf_prediction_set(qhat, predict_scores):
    prediction_set = []
    if predict_scores[0] >= (1-qhat):
        prediction_set.append((predict_scores[0], 0))
    if predict_scores[1] >= (1-qhat):
        prediction_set.append((predict_scores[1], 1))
    return prediction_set, len(prediction_set)

def compute_q_hat(predictions, labels, alpha, n):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    x = float(1)
    class_1 = np.array(predictions)
    class_0 = x-class_1
    true_class_prob = []
    conformal_scores = []

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
    print("quantile value: "+ str(qhat))
    return qhat

def CPC_classification(t_predictions, t_true_labels, quantile0, quantile1, quantile2, alpha0, alpha1, alpha2, index=0):
    # Make sure t_predictions is a list or array, otherwise convert it
    if isinstance(t_predictions, pd.Series) or isinstance(t_predictions, pd.DataFrame):
        t_predictions = t_predictions.tolist()

    res = []
    for t in t_predictions:
        t_prob = get_classes_prob(t)
        predicted_label = get_predicted_label(t, c_threshold)
        set0, size0 = compute_conf_prediction_set(quantile0, t_prob)
        set1, size1 = compute_conf_prediction_set(quantile1, t_prob)
        set2, size2 = compute_conf_prediction_set(quantile2, t_prob)
        tl = t_true_labels[t_predictions.index(t)]
        category = 0
        correct_c1 = -1
        correct_c2 = -1
        correct_c3 = -1
        correct_c4 = -1
        C1_class = 0
        C2_class = 0
        C3_class = 0
        C4_class = 0

        if size0 == 1: # & (set0[0][0] < (1-quantile1))
                category = 1
                if set0[0][1] == tl:
                    correct_c1 = 1
                else:
                    correct_c1 = 0
                if tl == 1:
                    C1_class = 1

        elif size1 == 1:
            #if (set1[0][0] < (1-quantile2)):
            category = 2
            if set1[0][1] == tl:
                correct_c2 = 1
            else:
                correct_c2 = 0
            if tl == 1:
                C2_class = 1

        elif size2 == 1:
            category = 3
            if set2[0][1] == tl:
                correct_c3 = 1
            else:
                correct_c3 = 0
            if tl == 1:
                C3_class = 1

        else:
            category = 4
            if size2 == 2:
                correct_c4 = 1
            elif size2 == 0:
                correct_c4 = 0
            elif set2[0][1] == tl:
                correct_c4 = 1
            else:
                correct_c4 = 0
            if tl == 1:
                C4_class = 1

        res.append((t_prob, tl, set0, size0, set1, size1, set2, size2, category, correct_c1, correct_c2, correct_c3, correct_c4, C1_class, C2_class, C3_class, C4_class))

    df = pd.DataFrame(res, columns=['softmax', 'true label','set0','size0','set1','size1', 'set2','size2', 'Category','correct_cat_1', 'correct_cat_2', 'correct_cat_3', 'correct_cat_4', 'Predicted_label_cat1', 'Predicted_label_cat2', 'Predicted_label_cat3', 'Predicted_label_cat4'])
    df.to_excel('Layered_CP_with_ALPHA_' + str(alpha0)+ '_'+str(alpha1) + '_'+str(alpha2)+ '_res_' + str(DATASET) + '_' + str(index) + '.xlsx')


    C1_total = len(df[df['Category'] == 1])
    C2_total = len(df[df['Category'] == 2])
    C3_total = len(df[df['Category'] == 3])
    C4_total = len(df[df['Category'] == 4])
    C1_validity = (float) (((df['correct_cat_1'] == 1)).sum() / C1_total) if C1_total > 0 else 0
    C2_validity = (float) (((df['correct_cat_2'] == 1)).sum() / C2_total) if C2_total > 0 else 0
    C3_validity = (float) (((df['correct_cat_3'] == 1)).sum() / C3_total) if C3_total > 0 else 0
    C4_validity = (float) (((df['correct_cat_4'] == 1)).sum() / C4_total) if C4_total > 0 else 0

    extract_CPC_statistics(len(t_predictions), C1_validity, C2_validity, C3_validity, C4_validity, len(df[df['size0'] == 1]), len(df[df['size1'] == 1]), len(df[df['size2'] == 1]), ((df['correct_cat_1'] == 1)).sum(), ((df['correct_cat_2'] == 1)).sum(), ((df['correct_cat_3'] == 1)).sum(), ((df['correct_cat_4'] == 1)).sum(), len(df[df['Category'] == 1]), len(df[df['Category'] == 2]), len(df[df['Category'] == 3]), len(df[df['Category'] == 4]),  ((df['Predicted_label_cat1'] == 1)).sum(), ((df['Predicted_label_cat2'] == 1)).sum(), ((df['Predicted_label_cat3'] == 1)).sum(), ((df['Predicted_label_cat4'] == 1)).sum())

    #extract_statistics(len(t_predictions), cp_validity, len(df[df['size'] == 0]), len(df[df['size'] == 1]), len(df[df['size'] == 2]), len(df[df['Conf pred check'] == 1]), len(df[df['Conf pred check'] == 0 ]), len(df[df['correct_size1_class'] == 1]), len(df[df['correct_size1_class'] == 0]), len(df[df['incorrect_size1_class'] == 1]),  len(df[df['incorrect_size1_class'] == 0]), alpha)


if __name__ == '__main__':
    alpha_0 = 0.05
    alpha_1 = 0.1
    alpha_2= 0.15
    # 1-alpha is the desired coverage
    #TODO: tweak the 'testing' variable below to obtain predictions on the validation or test set
    # 'testing' controls whether the predictions will be made on the validation or on the test set: True=> on the test set; False=> on the valid set
    testing = True

    if testing is True:
        save_file = f'{MODEL}/{DATASET}/CPC_with_{DATASET}_with_alpha_{alpha_0}_{alpha_1}_{alpha_2}_test_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename = f'{MODEL}/CPC_with_{DATASET}_test_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/test_set'
    else:
        save_file = f'{MODEL}/{DATASET}/CPC_with_{DATASET}_with_alpha_{alpha_0}_{alpha_1}_{alpha_2}_validation_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename = f'{MODEL}/CPC_with_{DATASET}_validation_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/validation_set'

    filesInCalib = sorted(os.listdir(folderCalib))
    filesInEval = sorted(os.listdir(folderEval))

    num_files = len(filesInCalib)
    print("Nr files inside Calib folder:", num_files)
    q_hats = []
    for i in range(num_files):
        file_calib = filesInCalib[i]
        file_eval = filesInEval[i]
        # step 1: use the trained model to make predictions on the calibration set
        print(f"Calibrating the model on Calib data - file {file_calib}")
        predictions, true_labels, calib_set_size = get_predictions(folderCalib + '/' + file_calib)

        # compute q-hat for the calibrated predictions using the original CP
        quantile_a0 = compute_q_hat(predictions, true_labels, alpha_0, calib_set_size)
        quantile_a1 = compute_q_hat(predictions, true_labels, alpha_1, calib_set_size)
        quantile_a2 = compute_q_hat(predictions, true_labels, alpha_2, calib_set_size)

        v_predictions, v_true_labels, valid_set_size = get_predictions(folderEval + '/' + file_eval)

        CPC_classification(v_predictions, v_true_labels, quantile_a0, quantile_a1, quantile_a2, alpha_0, alpha_1, alpha_2, i)
        file = open(save_file, "a")  # see main.py
        file.write(" {} dataset experiment nr: {}: ".format(DATASET, i) + "\n")
        file.write(" Conformal set size: " + str(calib_set_size) + "\n")
        file.write(" Q-hat with normal CP on alpha:{0:f}: {1:f} : ".format(alpha_0, quantile_a0) + "\n")
        file.write(" Q-hat with normal CP on alpha:{0:f}: {1:f} : ".format(alpha_1, quantile_a1) + "\n")
        file.write(" Q-hat with normal CP on alpha:{0:f}: {1:f}: ".format(alpha_2, quantile_a2) + "\n")
        file.write("------------------------------------------------------------ \n")
        file.close()

    write_CPC_statistis(filename)


