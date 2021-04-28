from pandas import read_excel, read_csv
from numpy import amax
from numpy import amin
from numpy import argmax
from numpy import array

from CBR_Model import CBR_Model
from Optimizer import Optimizer

def init(data, cols):
    location_diff = [
        1, 
        0.857142857,
        0.714285714,
        0.571428571,
        0.428571429,
        0.285714286,
        0.142857143,
        0
    ]

    max_diff = amax(data, axis=0) - amin(data, axis=0)
    maxdiff = {col: max_diff[i] for i, col in enumerate(cols)}
    fns = {
        'W2V': 
            lambda x, y: 1 - abs(x-y) / maxdiff.get('W2V', 1),
        'D2V':
            lambda x, y: 1 - abs(x-y) / maxdiff.get('D2V', 1),
        'cos_sim':
            lambda x, y: 1 - abs(x-y) / maxdiff.get('cos_sim', 1),
        'surface similarity':
            lambda x, y: 1 - abs(x-y) / maxdiff.get('surface similarity', 1),
        'paper sizein Bytes':
            lambda x, y: 1 - abs(x-y) / maxdiff.get('paper sizein Bytes', 1),
        'publication type pair; 1 same type;0 different types':
            lambda x, y: 1 if x == y else 0,
        'in which percentile (1 to 8) the citation occurs':
            lambda x, y: location_diff[int(abs(x-y)/1000)]
    }
    return CBR_Model(diff_fns=[fns.get(col, 0) for col in cols[:-1]])

def optimize(model, data, itermax=50, k=1, act_fn=round):   # round rounds .5 to zero
    weights = [model.W]
    acc, res = model.test_acc(data, k=k, act_fn=act_fn)
    accuracies = [acc]
    ress = [res]
    for _ in range(itermax):
        model.W = Optimizer.gradient_descent(model, data)
        weights.append(model.W)
        acc, res = model.test_acc(data, k=k, act_fn=act_fn)
        accuracies.append(acc)
        ress.append(ress)
    return weights, accuracies, ress

if __name__ == '__main__':
    # ---------- Hyper Parameters Start Here ----------

    # The path of Excel file.
    file_path = '100-case.xlsx'

    # Columns we are going to use. Put col Y at the end.
    cols = [
        'W2V', 
        'D2V',
        'cos_sim',
        'surface similarity',
        'paper sizein Bytes',
        'publication type pair; 1 same type;0 different types',
        'in which percentile (1 to 8) the citation occurs',
        'final decision: 1 is B, 0 is S'
    ]

    # Times Run
    times = 10

    # The max times GD will run.
    itermax = 50

    # Use k most similar cases for prediction.
    k = 1

    # ---------- Hyper Parameters End Here ----------

    # --- Load from Excel File ---
    excel = read_excel(file_path, header=1, usecols=range(9),engine='openpyxl')
    cos_sim = read_csv('../bert-based_explanation/cos_sim.csv')

    excel['cos_sim'] = cos_sim['cos_sim']
    # --- Data Cleaning ---
    data = excel[cols].values
    # --- Run T times ---
    total = 0
    total_f1 = 0
    for T in range(1, times+1):
        # --- Init CBR Model ---
        model = init(data, cols)
        # --- Optimize Model Weight ---
        w, acc, res= optimize(model, data, itermax=itermax, k=k)
        # --- Print Result ---
        # print to file each accuracy and its corresponding weights W
        with open('report_{}.txt'.format(T), 'w') as f:
            for i, accy in enumerate(acc):
                print('----- Iteration {} -----'.format(i), file=f)
                print('Weights:', file=f)
                print(w[i], file=f)
                print('Accuracy: {}'.format(accy), file=f)
                print('F1: {}'.format(res[i]), file=f)
        # print the maximum accuracy and its corrersponding weights
        print('----- {} -----'.format(T))
        index = argmax(array(acc))
        print('Weights:')
        print(w[index])
        print('Accuracy: {}'.format(acc[index]))
        print('F1: {}'.format(res[index]))
        total += acc[index]
        total_f1 += res[index]
    # --- Print the average accuracy ---
    avg = total / times
    print('Average Accuracy is {}.'.format(avg))
    avg_f1 = total_f1 / times
    print('Average F1 is {}.'.format(avg_f1))
