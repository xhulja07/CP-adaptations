from scipy import stats
import ast
import csv
import pandas as pd
import os

DATASET = 'os'
# Dataset values: 'os', 'qt'
MODEL = 'LApredict'
# Model values:['LApredict', 'DeepJIT', 'LApredict']

def write_staistics(all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test, headers):
    """ Method used to write the calculated statistics to CSV file.
    """
    # Validation FOlds
    filename=f"aggregated_statistics_validation_{DATASET}_{MODEL}.csv"
    new_headers = []
    p_values_cal_val = []
    p_values_uncal_val = []
    p_values_cal_test = []
    p_values_uncal_test = []
    for index, header in enumerate(headers):
        suffix_platt = "_platt"
        suffix_temp = "_temp"
        new_headers.append(header + suffix_platt)
        new_headers.append(header + suffix_temp)

    for p_value_cal, p_value_uncal in all_p_values_normality_val:
        p_values_cal_val.append(p_value_cal)
        p_values_uncal_val.append(p_value_uncal)

    for p_value_cal, p_value_uncal in all_p_values_normality_test:
        p_values_cal_test.append(p_value_cal)
        p_values_uncal_test.append(p_value_uncal)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_val)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_val)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_val)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_val)

    # Test data
    filename=f"aggregated_statistics_test_{DATASET}_{MODEL}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_test)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_test)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_test)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_test)


def read_input(filename):
    df = pd.read_excel(filename)
    return df


def calculate_significance(before_samples, after_samples, required_p_value=0.05):
    """
    Calculate significance for samples before and after treatment application (e.g., before and after calibration) either via t_test_paired or wilcoxon, depending on normality.

    Params:
        before_samples (numpy.arr)
        after_samples (numpy.arr)
        all_statistics_t_test (arr): array holding all repetitions, calculated values should be added here
        all_statistics_wilcoxon (arr): array holding all repetitions, calculated values should be added here
        all_p_values (arr): array holding tuples of p_values for normality
    """
    normal = True
    p_value_t_test = paired_t_test(before_samples, after_samples)
    p_value_wilcoxon = -1
    before_samples_normal = is_normal_distributed(after_samples, is_uncal=False)
    after_samples_normal = is_normal_distributed(before_samples, is_uncal=True)
    if not (before_samples_normal[1] and after_samples_normal[1]):
        print(f"[ERROR] not normally distributed")
        normal = False
        p_value_wilcoxon = wilcox_test(before_samples, after_samples)
        p_value_t_test = -1
    #all_statistics_t_test.append(p_value_t_test)
    #all_statistics_wilcoxon.append(p_value_wilcoxon)
    #all_p_values.append((uncalibrated_samples_normal, calibrated_samples_normal))
    return p_value_t_test, p_value_wilcoxon, after_samples_normal, before_samples_normal, normal


def statistic(x, axis):
    """
    Taken from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html
    """
    # Get only the `normaltest` statistic; ignore approximate p-value
    # still requires n >= 20
    return stats.normaltest(x, axis=axis).statistic


def is_normal_distributed_mc(samples, is_uncal, required_p_value=0.05):
    """
    Caclualte normality based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html#scipy.stats.monte_carlo_test, example from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html
    Functioning is similar to other func: H_0 = is normally distributed, p_value <= 0.05 significant rejection of H_0
    """
    calibration = "uncalibrated" if is_uncal else "calibrated"
    res = stats.monte_carlo_test(samples, stats.norm.rvs, statistic)
    print(f"Result = {res}")
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION MC] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION MC] for {calibration} p-value = {res.pvalue}")
    return (res.pvalue, True)


def is_normal_distributed_dagostino(samples, is_uncal, required_p_value=0.05):
    """
    Method used to determine if samples are drawn from a normal distribution.
    H_0_ = it is normally distributed
    Hence, a low p_value can be seen as evidence, that this sample distribution is not drawn from a normal distribution.
    Sufficient "high" p_value currently (randomly) set as 0.2.
    Noteworthy: only works properly for at least 50 samples.
    More infos https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html#hypothesis-normaltest
    """
    res = stats.normaltest(samples)
    calibration = "uncalibrated" if is_uncal else "calibrated"
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION] statistic for {calibration} distribution: {res.statistic}\np-value = {res.pvalue}")
    return (res.pvalue, True)

def is_normal_distributed(samples, is_uncal, required_p_value=0.05):
    if len(samples) >= 50:
        print(f"[NORMAL DISTRIBUTION] Sample size >= 50 using default normality for is_uncal = {is_uncal}")
        return is_normal_distributed_dagostino(samples, is_uncal, required_p_value)
    print(f"[NORMAL DISTRIBUTION] Sample size < 50 using monte carlo normality for is uncal = {is_uncal}")
    return is_normal_distributed_mc(samples, is_uncal=False, required_p_value=required_p_value)


def paired_t_test(samples_uncal, samples_cal):
    """
    Compute the paired t test for the uncalibrated and calibrated ECE scores.
    This assumes H_0_ = identical average expected values for both samples

    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    result = stats.ttest_rel(samples_uncal, samples_cal)
    print(f"[PAIRED T_TEST]] results = {result}\n with conf interval = {result.confidence_interval(0.95)}")

    return result.pvalue


def wilcox_test(uncalibrated_samples, calibrated_samples):
    """ TODO: Question of rounding for proper analysis -> else simply uncal - cal will be used. This might lead to some weird behaviour, as due to floating inacurracy the some numbers are treated (un)equally, that should not.
    H_0: samples are from same distribution
    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    result = stats.wilcoxon(uncalibrated_samples, calibrated_samples)
    print(f"[WILCOXON]] results = {result}\n")

    return result.pvalue


def load_data_metrics(file_path):
    df = pd.read_excel(file_path) # The file must contain the measured metric (precision and recall) before and after the treatment application, in different columns
    class_0_before = df['Class=0_precision_before'].tolist()
    class_1_before = df['Class=1_precision_before'].tolist()
    class_0_after = df['Class=0_precision_after'].tolist()
    class_1_after = df['Class=1_precision_after'].tolist()
    return class_0_before, class_1_before, class_0_after, class_1_after

if __name__ == '__main__':
    #CALIBRATION = 'Platt'
    # Calibration values:['Platt', 'Uncalib']
    required_p_value = 0.05
    save_file = f'statistical_significance_results_for_{MODEL}_on_{DATASET}.txt'
    file_path = f'{MODEL}/{DATASET}/statistical_analysis'  # Update with your actual file path
    filesInFolder = sorted(os.listdir(file_path))
    files_xls = [f for f in filesInFolder if f[-4:] == 'xlsx']
    num_files = len(files_xls)
    file = open(save_file, "a")  # see main.py
    file.write("\n Calculating statistical significance via T-test or Wilcoxon: \n")
    for i in range(num_files):
        filename = files_xls[i]
        print("File:", filename)
        class_0_before, class_1_before, class_0_after, class_1_after = load_data_metrics(file_path + '/' +filename)
        file.write("\n File name: {}".format(filename))
        all_statistics_t_test, all_statistics_wilcoxon, p_value_before, p_value_after, normally_distributed = calculate_significance(class_0_before, class_0_after)
        file.write("\n Stat. Tests for class=0:: \n \n")
        if all_statistics_t_test < required_p_value:
            file.write(f"\n [PAIRED T_TEST] Likely that the difference is statistically significant with p_value of= {all_statistics_t_test} \n")
        else:
            file.write(f"\n [PAIRED T_TEST] Likely that the difference is not statistically significant with p-value = {all_statistics_t_test} \n")

        if all_statistics_wilcoxon < required_p_value:
            file.write(f"\n [WILCOXON TEST] Likely that the difference is statistically significant with p_value of {all_statistics_wilcoxon} \n")
        else:
            file.write(f"\n [WILCOXON TEST] Likely that the difference is not statistically significant with p-value = {all_statistics_wilcoxon} \n")

        file.write(f"\n paired_t_test {all_statistics_t_test} \n, wilcoxon {all_statistics_wilcoxon} \n, Normality check for before {p_value_before} and after modification {p_value_after}, normally distributed = {normally_distributed} \n \n \n")

        all_statistics_t_test1, all_statistics_wilcoxon1, p_value_before1, p_value_after1, normally_distributed1 = calculate_significance(class_1_before, class_1_after)
        file.write("\n Stat. Tests for class=1::  \n")
        if all_statistics_t_test1 < required_p_value:
            file.write(
                f" \n [PAIRED T_TEST] Likely that the difference is statistically significant with p_value of= {all_statistics_t_test1} \n")
        else:
            file.write(
                f"\n[PAIRED T_TEST] Likely that the difference is not statistically significant with p-value = {all_statistics_t_test1} \n")

        if all_statistics_wilcoxon1 < required_p_value:
            file.write(
                f" \n [WILCOXON TEST] Likely that the difference is statistically significant with p_value of {all_statistics_wilcoxon1} \n")
        else:
            file.write(
                f"\n [WILCOXON TEST] Likely that the difference is not statistically significant with p-value = {all_statistics_wilcoxon1} \n")

        file.write(
            f"\n paired_t_test {all_statistics_t_test1} \n, wilcoxon {all_statistics_wilcoxon1} \n, Normality check for before {p_value_before1} and after modification {p_value_after1}, normally distributed = {normally_distributed1} \n")
        file.write("\n ----------------------------------------- \n")




