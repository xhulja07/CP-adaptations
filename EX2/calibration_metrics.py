from sklearn.calibration import calibration_curve as reliability_diagram
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from netcal.metrics import ECE

import csv
import numpy as np
# from conformal_prediction import apply_calibration, apply_conformal_prediction
import matplotlib.pyplot as plt



# we experiment with bin size "n_bins" = 15 and "n_bins" = 50

def _full_accuracy_rel_dia(labels, preds, true_preds, n_bins, iteration, dataset, interactive_binning=False, calibration=None, extract_diagram=False):
    # cf. https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/calibration.py#L915 (extended for correct accuracy)
    if interactive_binning:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(preds, quantiles * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], preds)

    bin_total = np.bincount(binids, minlength=len(bins))
    bin_sums = np.bincount(binids, weights=preds, minlength=len(bins))
    bin_true_pos_and_neg = np.bincount(binids, weights=true_preds, minlength=len(bins))

    nonzero = bin_total != 0
    confidence = bin_sums[nonzero] / bin_total[nonzero]
    accuracy = bin_true_pos_and_neg[nonzero] / bin_total[nonzero]

    if extract_diagram is True:
        plot_rel_dia(confidence, accuracy, n_bins, iteration, dataset, interactive_binning, False, calibration)

    return confidence, accuracy, bin_total


def plot_rel_dia(confidence, accuracy, bins, iteration, dataset, interactive_binning=False, scikit_impl=False,
                 calibration=None):
    # Plot the bar chart
    fig = plt.figure(figsize=(12, 6))
    bar_width = 0.01
    label = 'Accuracy'
    if scikit_impl: label = 'Fraction of fault-prone commits'
    plt.bar(confidence - bar_width / 2, accuracy, bar_width, color='b', alpha=0.6, label=label)

    # Plot the reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

    # Add labels and title
    plt.xlabel('Confidence')
    plt.ylabel(label)
    plt.title(
        f'Reliability Diagram for iteration {iteration} of {dataset} dataset with {len(confidence)} actual bins (interactive binning {interactive_binning}) calibration_{calibration}')
    plt.legend()

    # Show plot
    plt.grid()
    # print(f"\n ITeration = {iteration}")
    # for index, conf in enumerate(confidence):
    #     print(f"[Bin {index}] confidence={conf}")
    plt.savefig(
        f'../experiment_results_oversampled_2/Rel_diagram_iteration {iteration}_{dataset}_{bins} bins interactive binning {interactive_binning}_scikit_impl_{scikit_impl}_calibration_{calibration}.png')
    plt.close(fig)


def evaluate_reliability_diagram(labels, preds, true_preds, n_bins, iteration, dataset, interactive_binning=False):
    # cf. https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/calibration.py#L915 (extended for correct accuracy)
    if interactive_binning:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(preds, quantiles * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], preds)

    bin_total = np.bincount(binids, minlength=len(bins))
    bin_sums = np.bincount(binids, weights=preds, minlength=len(bins))
    bin_true = np.bincount(binids, weights=labels, minlength=len(bins))

    nonzero = bin_total != 0
    confidence = bin_sums[nonzero] / bin_total[nonzero]
    accuracy = bin_true[nonzero] / bin_total[nonzero]

    # if interactive_binning:
    #     accuracy, confidence = reliability_diagram(labels, preds, n_bins=bins, strategy='quantile')
    # else:
    #     accuracy, confidence = reliability_diagram(labels, preds, n_bins=bins, strategy='uniform')

    plot_rel_dia(confidence, accuracy, n_bins, iteration, dataset, interactive_binning, True)

    return confidence, accuracy, bin_total, bin_total[nonzero]


def calculate_ece(confidence, accuracy, verbose=False, sample_size=1):
    total_error = 0
    number_actual_bins = len(confidence)
    for index, confidence_bin in enumerate(confidence):
        error = confidence[index] - accuracy[index]
        # print(f"[ECE bin {index}] confidence_bin = {confidence_bin} with accuracy[index] = {accuracy[index]}")
        if error < 0: error *= -1
        total_error += error

    return total_error /number_actual_bins


def compute_ece_mce(p, l, b, interactive_binning= False):
    """
    Compute the Expected Calibration Error (ECE) score and MCE (max calibration error)
    Args:
    p (np.array): Array of prediction probabilities (size N).
    l (np.array): Array of true labels (size N).
    b (int): Number of bins for calibration.

    Returns:
    2 float-s: ECE score and MCE score
    """
    ece = 0.0
    N = len(p)
    mce=0
    # Discretize probabilities into bins
    if interactive_binning:
        quantiles = np.linspace(0, 1, b + 1)
        bin_edges = np.percentile(p, quantiles * 100)
    else:
        bin_edges = np.linspace(0, 1, b + 1)

    for i in range(b):
        # Find the indices of predictions falling within the current bin
        in_bin = np.where((p >= bin_edges[i]) & (p < bin_edges[i + 1]))[0]

        if len(in_bin) == 0:
            continue

        # Calculate the accuracy of the predictions in this bin
        correct = []
        for i in in_bin:
            if l[i] == 1:
                correct.append(i)

        accuracy = len(correct)/len(in_bin)

        # Calculate the average predicted probability in this bin
        avg_prob = np.mean(p[in_bin])

        # Compute the absolute difference between accuracy and average probability
        if mce < abs(accuracy - avg_prob):
            mce= abs(accuracy - avg_prob)
        ece += len(in_bin) / N * abs(accuracy - avg_prob)

    return ece, mce

def calculate_mce(confidence, accuracy):
    max_error = 0
    max_error_index = 0
    for index, confidence_bin in enumerate(confidence):
        error = confidence[index] - accuracy[index]
        if error < 0: error *= -1
        if error > max_error:
            max_error = error
            max_error_index = index

    return max_error


def calculate_brier_score(labels, preds):
    return brier_score_loss(labels, preds, pos_label=1.0)