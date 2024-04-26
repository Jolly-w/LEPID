from warnings import simplefilter
import pandas as pd
from macrocluster import *
from classify import *
from config import *
from utils import *
from scipy.spatial import distance
from stream_al.budget_manager.uncertainty_budget import EstimatedBudget
from stream_al.selection_strategies.ActiveLabeling import Random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

np.seterr(divide='ignore',invalid='ignore')
from sklearn.metrics import recall_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
def get_randomseed(random_state):
    return random_state.randint(2**31-1)
epsilon = 1e-10
simplefilter(action="ignore", category=FutureWarning)
original_data = pd.read_csv('E:\Experiment\datasets\SEA_abrupt.csv')

# data pre-processing
X_original = original_data.iloc[:,:-1]
y_original = original_data.iloc[:,-1]
X_orig = X_original.values
y_orig = y_original.values
X_train = X_orig[:1000]
X_test = X_orig[1000:]
y_train = y_orig[:1000]
y_test = y_orig[1000:]
original_train = original_data.iloc[:1000]
label_data = [x for _, x in original_train.groupby(original_train.iloc[:, -1])]
label_list = original_train.iloc[:, -1].unique()
label_list.sort()
data1 = label_data[0]
data2 = label_data[1]
data1_train = data1.iloc[:,:-1].values
data2_train = data2.iloc[:,:-1].values
model_clu = CluStream()
model_clu.fit(data1_train, label_list[0])
model_clu.fit(data2_train, label_list[1])
test_size = X_test.shape[0]
acc = []
correct_classifications = []
correct = 0
query_count = 0
j = 1
i = 1
theta = 0.1
####
thetai = 0.5
sampled = []
M = 34
# labeling  cost  budget
Budget_manager = EstimatedBudget(budget=0.1, w=100)
AL = Random()
drift_data = []
y_pred = []
y_true = []
y_random = []
acc_plot = []
Step = []
Imb_ratio = 0
times_step = []
while i < test_size:
    ex = {}
    ex['data'] = X_test[i,:]
    ex['label'] = y_test[i]
    y_true.append(ex['label'])
    x_cand = ex['data']
    y_cand = ex['label']
    CurrentTime = i
    model = model_clu.micro_clusters[:]
    p_label, mc_id, margin_x,max_label,second_label,entropy_c = classify(ex=ex['data'], model=model)
    y_pred.append(p_label)
    n_cluster = model[mc_id]
    n_center = n_cluster.Mc_center
    IC = distance.euclidean(x_cand,n_center)
    MCs = len(model)
    R = n_cluster.Mc_radius
    special_factor = n_cluster.Mc_special

    try:
        IL = IC/n_cluster.Mc_radius
    except ZeroDivisionError:
        IL = 0
    if Budget_manager.is_budget_left():
        if AL.labeling(theta=theta):
            sampled = [True]
            y_random.append(ex['label'])
            query_count += 1
            Budget_manager.update_budget(sampled=sampled)
            Imb_ratio = MaxMin_Ratio(y_random, p_label)
            if 0 < IL <= 0.5:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    n_cluster.W -= 1
                    if n_cluster.W < 0:
                        model.remove(n_cluster)
                        model_clu.creat(x=ex['data'],y=ex['label'],CurrentTime=CurrentTime,Mc_labeling=1,Mc_spe=2)
            elif 0.5 < IL <= 1:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    model_clu.break_cluster(X=ex['data'],y=ex['label'],cluster=n_cluster,IL=IL,CurrentTime=CurrentTime)
            else:
                model_clu.creat(x=ex['data'],y=ex['label'],CurrentTime=CurrentTime,Mc_labeling=1,Mc_spe=2)

        elif AL.special_cluster(special_factor=special_factor):
            sampled = [True]
            y_random.append(None)
            query_count += 1
            Budget_manager.update_budget(sampled=sampled)
            if Imb_ratio > thetai:
                n_cluster.Mc_special += Imb_ratio
            if 0 < IL <= 0.5:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    n_cluster.W -= 1
                    if n_cluster.W < 0:
                        model.remove(n_cluster)
                        model_clu.creat(x=ex['data'], y=ex['label'], CurrentTime=CurrentTime, Mc_labeling=1, Mc_spe=2)
            elif 0.5 < IL <= 1:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    model_clu.break_cluster(X=ex['data'], y=ex['label'], cluster=n_cluster, IL=IL,
                                            CurrentTime=CurrentTime)
            else:
                model_clu.creat(x=ex['data'], y=ex['label'], CurrentTime=CurrentTime, Mc_labeling=1, Mc_spe=2)


        elif AL.les(ent=entropy_c):
            sampled = [True]
            y_random.append(None)
            query_count += 1
            Budget_manager.update_budget(sampled=sampled)
            if 0 < IL <= 0.5:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    n_cluster.W -= 1
                    if n_cluster.W < 0:
                        model.remove(n_cluster)
                        model_clu.creat(x=ex['data'], y=ex['label'], CurrentTime=CurrentTime, Mc_labeling=1, Mc_spe=2)
            elif 0.5 < IL <= 1:
                if p_label == ex['label']:
                    model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=1)
                    n_cluster.W += 1
                else:
                    n_cluster.Mc_special += 1
                    model_clu.break_cluster(X=ex['data'], y=ex['label'], cluster=n_cluster, IL=IL,
                                            CurrentTime=CurrentTime)
            else:
                model_clu.creat(x=ex['data'], y=ex['label'], CurrentTime=CurrentTime, Mc_labeling=1, Mc_spe=2)

        else:
            sampled = [False]
            y_random.append(None)
            if IL < 1:
                model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=0)
            else:
                drift_data.append(ex['data'])
            Budget_manager.update_budget(sampled=sampled)
    else:
        sampled = [False]
        y_random.append(None)
        if IL < 1:
            model_clu.update_cluster(x=ex['data'], cluster=n_cluster, CurrentTime=CurrentTime, labeling=0)
        else:
            drift_data.append(ex['data'])
        Budget_manager.update_budget(sampled=sampled)
    # model_clu.update_model(CurrentTime=CurrentTime,y_random=y_random)
    model_clu.update_model(CurrentTime=CurrentTime)
    if p_label == ex['label']:
        correct += 1
        # model_clu.update_cluster()

    if i % 1000 == 0:
        Step.append(i)
        print('\n example no =', i)

    i += 1
    j += 1


    gmean = geometric_mean_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

print('\t Final gmean=', gmean)
print('\t Final recall=', recall)
print('\t Final F1=', f1)

