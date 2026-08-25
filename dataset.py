import torch
from torch.utils.data import IterableDataset
from torch.distributions import Beta
import random


class TabPFN_Dataset(IterableDataset):
    #TODO
    #dag generation should be examined
    #root node val generation sampling problem??
    # w and b sampling with log uniform??
   

    def __init__(self):
        super().__init__()



    '''
   Table 5: Overview of our prior hyperparameter distribution. For many features we use a Log Uniform
    distribution with truncated normal noise, which we refer to as TNLU(h|µ, ˇ µ, min, round ˆ ). We
    sample from it by first sampling mean µ and standard deviation σ from µ, σ ∼ LogUniform(ˇµ, µˆ)
    and then sampling from the resulting truncated normal distribution v ∼ TruncNormal(µ, σ2
    , a =
    0, b = inf). v is rounded to the closest integer, if round is set. The final sampled value then is
    h = v + min.
    '''
   

    def sample_from_log_uniform(self, min_mean, max_mean):
        
        u = torch.rand(1)
        val = min_mean * ((max_mean / min_mean) ** u)

        return val
    
    def tnlu(self,min_mean, max_mean, min_val, round = True):

        mean = self.sample_from_log_uniform(min_mean,max_mean)
        std = self.sample_from_log_uniform(min_mean,max_mean)

        val = -1
        while val < 0: # I dont wanna do that but...
            val = torch.normal(mean=mean,std=std)

        
        val = val + min_val

        if round:
            val = torch.round(val).long()

        return val

    def generate_random_dag(self):
        
        #CONSTANTS

        MLP_WEIGHT_DROPOUT_BETA_MIN = 0.1
        MLP_WEIGHT_DROPOUT_BETA_MAX = 5

        #===============================================
        # MLP WEIGHT DROPOUT 
        # Dropout ratio is sampled from beta distribution, which a,b sampled uniformly between 1 and 5
        alpha = (MLP_WEIGHT_DROPOUT_BETA_MIN - MLP_WEIGHT_DROPOUT_BETA_MAX) * torch.rand(1) + MLP_WEIGHT_DROPOUT_BETA_MAX
        beta_param = (MLP_WEIGHT_DROPOUT_BETA_MIN - MLP_WEIGHT_DROPOUT_BETA_MAX) * torch.rand(1) + MLP_WEIGHT_DROPOUT_BETA_MAX

        mlp_dropout_threshold = Beta(alpha,beta_param).sample() * 0.9
        #===============================================


        layer_count = self.tnlu(1,6,2,round=True)

        input_nodes_size = self.tnlu(1,12,1,round=True).item()
        layer_sizes = [input_nodes_size]

        #BLOCKWISE DROPOUT SHOULD BE IMPLEMENTED FOR WHOLE NETWORK.
        blockwise_dropout = (torch.rand(1) > 0.5).long()

        for i in range(layer_count - 1):
            ls = self.tnlu(5,130,4,round=True).item()
            layer_sizes.append(ls)

        #adjacency matrix yapcaz.
        
        A_size = sum(layer_sizes)
        A = torch.zeros((A_size,A_size))

        cumsum = torch.cumsum(torch.tensor(layer_sizes), dim=0)
        cumsum = torch.concatenate((torch.tensor([0]), cumsum))

        for l in range(layer_count - 1):

            dst_start = cumsum[l+1]
            dst_end = cumsum[l+2]

            src_start = cumsum[l]
            src_end = cumsum[l+1]

            src_mid = (src_start + src_end) // 2
            dst_mid = (dst_start + dst_end) // 2

            # BLOCKWISE DROPOUT
            if blockwise_dropout and (dst_end - dst_start) > 1 and (src_end - src_start) > 1:
                A[dst_start:dst_mid,src_start:src_mid] = 1
                A[dst_mid:dst_end,src_mid: src_end] = 1 
            else:
                A[dst_start:dst_end,src_start:src_end] = 1

        
        #MLP WEIGHT DROPOUT
        mask = (torch.rand_like(A) > mlp_dropout_threshold).long()
        A = A * mask 

        return A, layer_sizes

    
    def generate_data_from_dag(self, dag,layer_sizes, num_samples,):
        
        activation_func = self.random_activation_func()
         #We sample one activation function a per dataset from {T anh, LeakyReLU, ELU, Identity} from paper TabPFN

        layer_sizes_cumsum = torch.cumsum(torch.tensor(layer_sizes), dim=0)
        layer_sizes_cumsum = torch.concatenate((torch.tensor([0]), layer_sizes_cumsum))

        columns = []
        for i in range(dag.shape[0]):

            parent_matrix = None
            if torch.sum(dag[i,:]) == 0:
                #determine root value.
                root_column = torch.randn(num_samples, 1)# we should concat in axis = 1 in the future..
                columns.append(root_column)
            else:

                parent_indices = torch.where(dag[i,:] == 1)[0] # [0] returns true ones.
                parent_columns = [columns[idx] for idx in parent_indices]

                parent_matrix = torch.concat(parent_columns, axis=1)

                input_size = parent_matrix.shape[1] # girdi feature sayısı

                w_std = self.tnlu(min_mean=0.01, max_mean=10.0, min_val=0.0, round=False) 
                W = torch.randn(*(input_size,1)) * w_std

                #Gaussian Noise Std. TNLU 0.3 0.0001 False 0.0
                noise_std = self.tnlu(min_mean=0.0001, max_mean=0.3, min_val=0.0, round=False)
                noise = torch.randn((num_samples,1)) * noise_std #aka epsilon acts as random noise-like
                
                column = activation_func(parent_matrix @ W + noise)
                columns.append(column)

        #Choose Zy
        sample_y_from_last_layer = (torch.rand(1) > 0.5).long()

        zy_node_index = None

        if sample_y_from_last_layer:
            last_layer_end = layer_sizes_cumsum[-1]
            last_layer_start = layer_sizes_cumsum[-2]

            zy_node_index = torch.randint(last_layer_start, last_layer_end, (1,)).item()
        else:
            layers_end = layer_sizes_cumsum[-1]
            layers_start = 0

            zy_node_index = torch.randint(layers_start, layers_end,(1,)).item()

        #Choose X nodes
        sample_feature_nodes_blockwise = (torch.rand(1) > 0.5).long()
        keep_SCM_feature_order = (torch.rand(1) > 0.5).long()


        total_nodes = layer_sizes_cumsum[-1].item()
        candidate_indices = [i for i in range(total_nodes) if i != zy_node_index]#excluding zy.
        total_feature_num = torch.randint(2, len(candidate_indices) + 1, (1,)).item()

        X_indices = []
        #Blockwise choice. burayı yazması biraz zorladı.
        if sample_feature_nodes_blockwise:


            residual_feature_num = total_feature_num
            
            for l,layer_size in enumerate(layer_sizes):
                if residual_feature_num == 0:
                    break

                start = layer_sizes_cumsum[l]
                end = layer_sizes_cumsum[l+1]

                block_feature_num = torch.randint(1,min(residual_feature_num,layer_size) + 1,(1,)).item()

                start = torch.randint(start, end - block_feature_num + 1, (1,)).item()
                # Alınan dilim:
                chosen_nodes = list(range(start, start + block_feature_num))
                X_indices.extend(chosen_nodes)# seçilen indisleri ekledik.
                residual_feature_num = residual_feature_num - block_feature_num
        else: # directly chose random k features.
            X_indices = random.sample(candidate_indices, total_feature_num)

        
        if not keep_SCM_feature_order:
            random.shuffle(X_indices)
        
        Zy = columns[zy_node_index]
        

        selected_columns = [columns[i] for i in X_indices]
        X = torch.cat(selected_columns,dim=1)

        return X, Zy


    def random_activation_func(self):

        def identity(x):
            return x
        # 2. Tanh (Hiperbolik Tanjant)
        def tanh(x):
            return torch.tanh(x)
        # 3. Leaky ReLU
        def leaky_relu(x, alpha=0.01):
            return torch.where(x > 0, x, alpha * x)
        # 4. ELU (Exponential Linear Unit)
        def elu(x, alpha=1.0):
            return torch.where(x > 0, x, alpha * (torch.exp(torch.clip(x, -50, 50)) - 1))
            
        choice = torch.randint(0, 4, (1,)).item()#random uniform choice from activation funcs.
        func = identity

        if choice == 0:
            func = identity
        elif choice == 1:
            func = tanh
        elif choice == 2:
            func = leaky_relu
        elif choice == 3:
            func = elu

        return func
        


    def __iter__(self):



        return 


ds = TabPFN_Dataset()
# 1. Çizgeyi ve katman boyutlarını üret
A, layer_sizes = ds.generate_random_dag()
# 2. 100 satırlık veriyi üret
X, Zy = ds.generate_data_from_dag(A, layer_sizes, num_samples=100)
print("A Adjacency Shape:", A.shape)
print("X Feature Matrix Shape:", X.shape)
print("Zy Target Shape:", Zy.shape)


        
