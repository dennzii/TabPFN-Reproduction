import torch
from torch.utils.data import IterableDataset
from torch.distributions import Beta


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

        return A

    
    def generate_data_from_dag(self, dag, num_samples):
        
        activation_func = self.random_activation_func()
         #We sample one activation function a per dataset from {T anh, LeakyReLU, ELU, Identity} from paper TabPFN

        columns = []
        for i in range(dag.shape[0]):

            parent_matrix = None
            if torch.sum(dag[i,:]) == 0:
                #determine root value.
                root_column = torch.random.randn(num_samples, 1)# we should concat in axis = 1 in the future..
                columns.append(root_column)
            else:
                parent_indices = torch.where(dag[i,:] == 1)[0] # [0] returns true ones.
                parent_columns = [columns[idx] for idx in parent_indices]

                parent_matrix = torch.concat(parent_columns, axis=1)

                itorchut_size = parent_matrix.shape[1] # girdi feature sayısı

                
                
                W = torch.random.randn(*(itorchut_size,1))
                b = torch.random.randn(1) #aka epsilon acts as random noise-like
                

                column = activation_func(parent_matrix @ W + b)

                columns.append(column)
        
        return columns


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
            
        choice = torch.random.choice(4)#random uniform choice from activation funcs.
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

A = ds.generate_random_dag()

#threshold=float('inf') #-> Hiçbir elemanı '...' ile gizleme, hepsini bas
# linewidth=200 -> Satırları erkenden alt satıra kırma
torch.set_printoptions(threshold=float('inf'), linewidth=200)

print(A.int())



        
