from property_prediction.data_utils import TaskDataLoader
import networkx as nx
from pysmiles import read_smiles

task = 'FreeSolv'
path = '../datasets/{}.csv'.format(task)

data_loader = TaskDataLoader(task, path)
smiles_list, y = data_loader.load_property_data()

indices = []

for i in range(len(smiles_list)):
    graph = read_smiles(smiles_list[i])
    number_of_nodes = nx.Graph.number_of_nodes(graph)
    print('number of nodes for index ', i, ' is: ', number_of_nodes)
    if number_of_nodes == 1:
        indices.append(i)
        print(smiles_list[i])

print(indices)