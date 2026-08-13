import torch
from torch.utils.data import IterableDataset
import numpy as np



class TabPFN_Dataset(IterableDataset):




    def __init__(self):
        super().__init__()




    def generate_random_dag(self, column_count = (2,100), thresh_interv= (0.1,0.9)):

        column_count = np.random.randint(*column_count)
        
        threshold = np.random.uniform(*thresh_interv)
        adj_matrix = (np.random.rand(column_count,column_count) > threshold).astype(int)
        dag = np.tril(adj_matrix,k=-1)# do not include main diagonal
        
        return dag
    
    def generate_from_dag(self, dag, num_samples):
        

        columns = []
        for i in range(dag.shape[0]):


            if np.sum(dag[i,:]) == 0:
                #determine root value.
                root_column = np.random.randn(num_samples, 1)# we should concat in axis = 1 in the future..
                columns[i] = root_column
            else:
                





        
