import torch
from torch.utils.data import IterableDataset
from torch.distributions import Beta
from torch.nn.functional import leaky_relu,elu,tanh
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import TabPFN_Dataset
from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
from google.colab import drive
from tqdm import tqdm

class Transformer_TabPFN(nn.Module):


    """
    We considered only PFN Transformers with 12 layers, embeddings size 512, hidden size 1024 in
    feed-forward layers, and 4-head attention. We used the Adam optimizer (Kingma and Ba, 2015) with
    linear-warmup and cosine annealing (Loshchilov and Hutter, 2017). For each training we tested a set
    of 3 learning rates, {.001, .0003, .0001}, and used the one with the lowest final training loss. The
    resulting model contains 25.82 M parameters.
    """
    

    def __init__(self,d_model = 512, num_layers = 12, num_heads = 4,hidden_size_ff = 1024, max_features = 100, num_classes = 10):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_features = max_features
        self.hidden_size_ff = hidden_size_ff
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

         # N, (features + 1) @( features +1),d_model
        self.embedding = nn.Linear(self.max_features + 1,self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size_ff,
            batch_first=True,
            activation='gelu'
        )

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.num_layers)
        
        self.classifier = nn.Linear(self.d_model,self.num_classes)


    def forward(self,x,n_train):

        attn_mask = torch.zeros((x.shape[1],x.shape[1]),device=x.device)
        attn_mask[:,n_train:] = float('-inf')

        x = self.embedding(x)
        x = self.encoder(x,mask = attn_mask)
        

        x_test = x[:,n_train:,:]#only compute the representation of test samples.

        out = self.classifier(x_test)

        return out



def train(LR=1e-4,TOTAL_STEPS=200_000):


    
    #drive.mount('/content/drive')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is being used.")
    
    model = Transformer_TabPFN().to(device)
    print("initialized model")

    progress_bar = tqdm(total=TOTAL_STEPS, desc="TabPFN Training")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(),lr=LR)

    scheduler = get_cosine_schedule_with_warmup(
                                                optimizer,
                                                num_warmup_steps= 3000,        # Isınma adım sayısı
                                                num_training_steps=TOTAL_STEPS    # Toplam eğitim adım sayısı
                                                )

    ds = TabPFN_Dataset()
    dataloader = DataLoader(ds, batch_size=None)#bs none demek batch dimension direkt yok demek.

    
    step_num = 0

    for X_train, y_train, X_test, y_test in dataloader:
        step_num += 1
        optimizer.zero_grad()

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        
        training_samples = torch.concatenate((X_train,y_train),dim=1)
        n_train = training_samples.shape[0]

        test_samples = torch.concatenate((X_test,torch.zeros((X_test.shape[0],1), device=device)),dim=1)

        x_full = torch.cat((training_samples,test_samples),dim=0)

        x_full = x_full.unsqueeze(0)


        logits = model(x=x_full,n_train=n_train)

        loss = criterion(logits.squeeze(0),y_test.squeeze(-1).long())
        loss.backward()

        optimizer.step()
        scheduler.step()
        

        
        running_loss = loss.item() if step_num == 1 else (0.95 * running_loss + 0.05 * loss.item())
        progress_bar.update(1)
        if step_num % 100 == 0:
            progress_bar.set_postfix({
                "Loss": f"{running_loss:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        if step_num % 10000 == 0:
            torch.save({
                'step': step_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"/content/drive/MyDrive/tabpfn_step_{step_num}.pt")
            print(f"--> Checkpoint saved at step {step_num}")
    
train()