from property_prediction.predict_with_shortestpath import main as predict
import os
import re

directory = os.fsencode('FlowMO-graphs/datasets')

GRAPH_KERNELS = {'CW', 'MK', 'RW', 'SP', 'SSP', 'T', 'PUTH', 'WL'}


for graph_kernel in GRAPH_KERNELS:
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        m = re.search('(.+?).csv', str(filename))
        task = m.group(1)
        path = 'FlowMO-graphs/datasets/', str(filename)
        predict(path, task, 1, 0.2, True, graph_kernel)
        ###code to save results in array or something
