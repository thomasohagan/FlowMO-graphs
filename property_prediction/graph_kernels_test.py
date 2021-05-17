from property_prediction.predict_with_graph_kernel import main as predict
import os
import re

directory = os.fsencode('FlowMO-graphs/datasets')

GRAPH_KERNELS = {'CW', 'MK', 'SP', 'SSP', 'T', 'PUTH', 'WL'}


for graph_kernel in GRAPH_KERNELS:
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        m = re.search('(.+?).csv', str(filename))
        task = m.group(1)
        path = 'FlowMO-graphs/datasets/', str(filename)
        predict(path=path, task=task, n_trials=1, test_set_size=0.2, use_rmse_conf=True, kernel=graph_kernel, N=20)
