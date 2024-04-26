import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

def classify(ex, model):
    ex = np.array(ex)
    distances = np.array([np.linalg.norm(obj.Mc_center - ex) for obj in model])
    nearest_indices = np.argsort(distances)[:10]
    distances_list = distances[nearest_indices]
    nearest_labels = [model[i].Mc_label for i in nearest_indices]
    label_counts = np.bincount(nearest_labels)
    label_probs = label_counts / 10
    entropy_value = entropy(label_probs, base=2)
    label_min_indices = {}
    for i, label in enumerate(nearest_labels):
        if label not in label_min_indices:
            label_min_indices[label] = i
    label_ex = {}
    for label in set(nearest_labels):
        label_count = Counter(nearest_labels)[label]
        label_distances = distances_list
        # label_mcd = min(label_distances)
        min_indices = label_min_indices.get(label, float('inf'))
        label_mcd = distances_list[min_indices]
        label_purity = label_count / len(nearest_labels)
        label_ex[label] = label_purity / label_mcd

    if len(label_ex) == 1 and isinstance(next(iter(label_ex.values())), (int, float)):
        max_label, max_value = list(label_ex.items())[0]
        margin_x = max_value
        second_label = max_label
        final_label = max_label
    else:
        sorted_items = sorted(label_ex.items(), key=lambda x: x[1], reverse=True)
        max_value = sorted_items[0][1]
        max_label = [k for k, v in label_ex.items() if v == sorted_items[0][1]]
        second_max_value = sorted_items[1][1]
        second_label = [k for k, v in label_ex.items() if v == second_max_value]
        margin_x = max_value - second_max_value
        final_label = max_label[0]

    nearest = label_min_indices.get(final_label, float('inf'))
    min_idx = nearest_indices[nearest]
    return final_label,min_idx,margin_x,max_label,second_label,entropy_value
