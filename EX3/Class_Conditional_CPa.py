import numpy as np
import pandas as pd
import os
import sys

alpha = 0.05 # defines the error of CP (1-alpha)
c_threshold= 0.5 # defines the classification threshold used by the CL-FP models, default = 0.5, adjust after performing threshold optimisation
DATASET = 'os'
# DATASET variable takes values 'os' or 'qt'
MODEL = 'LApredict'
# CL-FP MODELs: DeepJIT, CodeBERT4JIT, LApredict

predictions_length_cc = []
defective_labels = []
CC0_nr_sets1 = []
CC1_nr_sets1 = []
CC0_validity_c1 = []
CC1_validity_c1 = []
CC0_correct = []
CC1_correct = []

def extract_cc_statistics(length, nr_labels_c1, cc0_validity_c1, nr_set_1_c0, correct_c0, cc1_validity_c1, nr_set_1_c1 , correct_c1):
    predictions_length_cc.append(length)
    defective_labels.append(nr_labels_c1)
    CC0_validity_c1.append(cc0_validity_c1)
    CC0_nr_sets1.append(nr_set_1_c0)
    CC0_correct.append(correct_c0)
    CC1_validity_c1.append(cc1_validity_c1)
    CC1_nr_sets1.append(nr_set_1_c1)
    CC1_correct.append(correct_c1)

def write_statistics(filename):

    df = pd.DataFrame({
        'nr_instances': predictions_length_cc,
        'nr_faulty_instances': defective_labels,
        'Class0_validity': CC0_validity_c1,
        'class0_size_1_sets': CC0_nr_sets1,
        'class0_correct': CC0_correct,
        'Class1_validity': CC1_validity_c1,
        'class1_size_1_sets': CC1_nr_sets1,
        'class1_correct': CC1_correct,
    })
    df.to_excel(filename, index=False)

def compute_class_conditional_q_hat(predictions, labels, alpha):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = np.array(predictions)
    class_0 = 1 - class_1
    conformal_scores_0 = []
    conformal_scores_1 = []

    for i in range(len(predictions)):
        if labels[i] == 0:
            pred_score = class_0[i]
            conformal_scores_0.append(1 - pred_score)
        else:
            pred_score = class_1[i]
            conformal_scores_1.append(1 - pred_score)

    n_0 = len(conformal_scores_0)
    n_1 = len(conformal_scores_1)

    #compute quantile
    q_level_0 = np.ceil((n_0 + 1) * (1 - alpha)) / n_0
    qhat_0 = np.quantile(conformal_scores_0, q_level_0, method='higher')

    q_level_1 = np.ceil((n_1 + 1) * (1 - alpha)) / n_1
    qhat_1 = np.quantile(conformal_scores_1, q_level_1, method='higher')

    return qhat_0, qhat_1

def comput_set_correctness(set_0, set_1, tl):
    correct_0 = -1
    correct_1 = -1
    if (len(set_0) != 0 and tl == 0):
        correct_0 = True
    elif (len(set_0) != 0 and tl == 1):
        correct_0 = -1
    else:
        correct_0 = False

    if (len(set_1) != 0 and tl == 1):
        # or (len(set_1) == 0 and tl == 0)
        correct_1 = True
    elif (len(set_1) != 0 and tl == 0):
        correct_1 = -1
    else:
        correct_1 = False

    return correct_0, correct_1

def class_cond_CP(t_predictions, t_true_labels, quantile_0, quantile_1, alpha, c_threshold, i=0):
    if isinstance(t_predictions, pd.Series) or isinstance(t_predictions, pd.DataFrame):
        t_predictions = t_predictions.tolist()
    cc_res = []
    for tp in t_predictions:
        cc_t_prob = get_classes_prob(tp)
        set_0, set_1, empty_sets = compute_class_cond_CP_set(quantile_0, quantile_1, cc_t_prob)
        # computes m_CC-CP: class conditional CP that classifies instances as either Clean or Fault-prone, but not both
        set_0_sp, set_1_sp = compute_class_cond_CP_set_single_class(quantile_0, quantile_1, cc_t_prob, c_threshold)
        tl = t_true_labels[t_predictions.index(tp)]

        # compute class-specific precision, i.e., how many of the sets contain the correct label
        correct_0, correct_1 = comput_set_correctness(set_0, set_1, tl)
        correct_0_sp, correct_1_sp = comput_set_correctness(set_0_sp, set_1_sp, tl)

        cc_res.append((cc_t_prob, tl, set_0, len(set_0), correct_0, set_1, len(set_1), correct_1, set_0_sp, len(set_0_sp), correct_0_sp, set_1_sp, len(set_1_sp), correct_1_sp))
    cc_df = pd.DataFrame(cc_res, columns=['softmax','true label', 'set_0','nr_set_0','correct_0', 'set_1', 'nr_set_1', 'correct_1', 'set_0_single_pred','nr_set_0_sp','correct_0_sp', 'set_1_single_pred', 'nr_set_1_sp', 'correct_1_sp'])
    cc_df.to_excel('CC-CPa_with_ALPHA' + str(alpha) + '_on_' + str(MODEL)+ '_with_' + str(DATASET)+ '_iteration_' +str(i) + '.xlsx')
    cp_validity_c0 = (float)(len(cc_df[cc_df['correct_0'] == True]) / len(cc_df[cc_df['nr_set_0'] ==1]))
    cp_validity_c1 = (float)(len(cc_df[cc_df['correct_1'] == True]) / len(cc_df[cc_df['nr_set_1'] ==1]))
    cp_validity_c0_sp = (float)(len(cc_df[cc_df['correct_0_sp'] == True]) / len(cc_df[cc_df['nr_set_0_sp'] == 1]))
    cp_validity_c1_sp = (float)(len(cc_df[cc_df['correct_1_sp'] == True]) / len(cc_df[cc_df['nr_set_1_sp'] == 1]))
    label_defect = len(cc_df[cc_df['true label'] == 1])

    #extract_cc_statistics(len(t_predictions), label_defect, cp_validity_c0, len(cc_df[cc_df['nr_set_0'] ==1]), len(cc_df[cc_df['correct_0'] == True]), cp_validity_c1, len(cc_df[cc_df['nr_set_1'] ==1]), len(cc_df[cc_df['correct_1'] == True]), cp_validity_c0_sp, len(cc_df[cc_df['nr_set_0_sp'] == 1]), len(cc_df[cc_df['correct_0_sp'] == True]), cp_validity_c1_sp, len(cc_df[cc_df['nr_set_1_sp'] == 1]), len(cc_df[cc_df['correct_1_sp'] == True]))
    extract_cc_statistics(len(t_predictions), label_defect, cp_validity_c0, len(cc_df[cc_df['nr_set_0'] ==1]), len(cc_df[cc_df['correct_0'] == True]), cp_validity_c1, len(cc_df[cc_df['nr_set_1'] ==1]), len(cc_df[cc_df['correct_1'] == True]))


def get_classes_prob(prediction):
    # for each instance, make a prediction and get the softmax score of the true label of that instance
    class_1 = prediction
    class_0 = 1 - class_1
    return np.array([class_0, class_1])

def get_predicted_label(predict_scores, c_threshold):
    if (predict_scores[1] >= c_threshold):
        return 1
    else:
        return 0

def compute_class_cond_CP_set(qhat_0, qhat_1, predict_scores):
    #class_conditional_CP_set - computes prediction sets for each class separately
    #this version allows both labels to be flagged as potentially correct, if each of the labels surpasses the respective q-hat for the class label
    prediction_set_0 = []
    prediction_set_1 = []
    set = 0

    if predict_scores[0] >= (1-qhat_0):
        prediction_set_0.append((predict_scores[0], 0))
        set = set +1
    if predict_scores[1] >= (1-qhat_1):
        prediction_set_1.append((predict_scores[1], 1))
        set = set + 1

    return prediction_set_0, prediction_set_1, set


def compute_class_cond_CP_set_single_class(qhat_0, qhat_1, predict_scores, c_threshold):
    "CC-CPa: This function is similar to the compute_class_cond_CP_set() with the only difference that it does not compute prediction sets for both possible labels but only for the label that is predicted"
    # this version allows only one of the possible labels to be flagged as potentially correct, prioritising the minority class
    prediction_set_0 = []
    prediction_set_1 = []
    c0 = predict_scores[0]
    c1 = predict_scores[1]
    if (c1 >= c_threshold):
        if (c1 >= (1 - qhat_1)):
            print("prediction probability for class = 1: ", c1)
            prediction_set_1.append((c1, 1))
        else:
            print("prediction probability for class = 1 DOES NOT SURPASS THRESHOLD ", c1)

    elif (c0 >= c_threshold):
           if c0 >= (1 - qhat_0):
                prediction_set_0.append((c0, 0))
           else:
               print("prediction probability for class = 0 DOES NOT SURPASS THRESHOLD ", c0)
    return prediction_set_0, prediction_set_1

def get_predictions(file_path):
    df = pd.read_excel(file_path)
    # Extract each column into separate lists (excluding the 'index' column)
    #input_list = df['Input'].tolist()
    true_label_list = df['True label'].tolist()
    predicted_prob = df['Predicted prob'].tolist()
    return predicted_prob, true_label_list, len(true_label_list)


if __name__ == '__main__':
    # update calib size to the nr of rows in the file
    #calib_set_size = 1000  # number of calibration instances
    alpha = 0.05  # 1-alpha is the desired coverage: in {0.05 , 0.10, 0.15}
    testing = True  # 'testing' controls whether the predictions will be made on the validation or on the test set: True=> on the test set; False=> on the valid set

    if testing is True:
        save_file = f'{MODEL}/{DATASET}/CC_CPa_with_{DATASET}_test_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename = f'{MODEL}/CC_CPa_with_{DATASET}_alpha_{alpha}_test_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/test_set'
    else:
        save_file = f'{MODEL}/{DATASET}/CC_CPa_with_{DATASET}_validation_set'
        #Ex 1- applying CPa on the predictions of the uncalibrated CL-FP models
        filename = f'{MODEL}/CC-CPa_with_{DATASET}_alpha_{alpha}_validation_set.xlsx'
        folderCalib = f'{MODEL}/{DATASET}/calibration_set'
        folderEval = f'{MODEL}/{DATASET}/validation_set'

    filesInCalib = sorted(os.listdir(folderCalib))
    filesInEval = sorted(os.listdir(folderEval))
    #print("Calibration folder", folderCalib)
    #print("Test folder", folderEval)
    num_files = len(filesInCalib)
    #print("Nr files inside Calib folder:", num_files)
    #Ex 1- applying CC-CPa on the predictions of the calibrated CL-FP models
    for i in range(num_files):
        print("Run uncalibrated nr:", i)
        file_calib = filesInCalib[i]
        file_eval = filesInEval[i]
        # step 1: use the trained model to make predictions on the calibration set
        print(f"Calibrating the model on Calib data - file {file_calib}")

        predictions, true_labels, calib_set_size = get_predictions(folderCalib + '/' + file_calib)

        # step 2: compute the empirical quantiles (q_hat for class 0 and 1) using the calibration set (predictions and true labels)
        qhat_0, qhat_1 = compute_class_conditional_q_hat(predictions, true_labels, alpha)

        # step 3: use the trained model to make predictions on the test data, or the validation data
        if testing is True:
            print(f"Evaluating the CP model on Test data - file {file_eval}")
        else:
            print(f"Evaluating the CP model on Valid data - file {file_eval}")

        t_predictions, t_true_labels, test_set_size = get_predictions(folderEval + '/' + file_eval)

        # step 4: apply class-conditional conformal prediction (CC-CPa) on the validation/test set predictions
        class_cond_CP(t_predictions, t_true_labels, qhat_0, qhat_1, alpha, c_threshold, i)


        # save data
        file = open(save_file, "a")  # see main.py
        file.write(" {} dataset experiment nr: {}: ".format(DATASET, i) + "\n")
        file.write(" Conformal set size: " + str(calib_set_size) + "\n")
        file.write(" Evaluation set size: " + str(test_set_size) + "\n")
        file.write(
            " qhat_0: {0:f}:, qhat_1: {1:f}: ".format(qhat_0, qhat_1) + "\n")
        file.write(" Alpha_0: {0:f}: ".format(alpha,) + "\n")
        file.write("------------------------------------------------------------ \n")
        file.close()

    write_statistics(filename)

