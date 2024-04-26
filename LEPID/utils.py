import pandas as pd
import numpy as np
# multiclass imbalance ratios

def calc_imby(y, y_random):
    labelnumy = y_random.count(y)
    valid_label = len([l for l in y_random if l is not None])
    C = len(set([l for l in y_random if l is not None]))
    if C == 0:
        return 1
    else:
        imby = labelnumy / ((valid_label) / C)
        return imby

def count_not_none(lst):
    count = 0
    for item in lst:
        if item is not None:
            count += 1
    return count
def print_percentages(lst):
    count = count_not_none(lst)
    for item in set(lst):
        if item is not None:
            percentage = round(100 * lst.count(item) / count, 2)
            print(f"{item}: {percentage}%")


def MaxMin_Ratio(y_random, y0):
    counts = {}
    max_count = 0
    y1_count = 0
    y0_count = 0

    for y in y_random:
        if y is None:
            continue

        if y == y0:
            y0_count += 1
        else:
            if y in counts:
                counts[y] += 1
            else:
                counts[y] = 1
            if counts[y] > max_count:
                max_count = counts[y]
                y1_count = counts[y]

    Ra = y0_count / (y1_count + y0_count)
    Ratio = 1 - Ra

    return Ratio





