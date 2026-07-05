import numpy as np
import pandas as pd
import os

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
