import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader


from sklvq import GLVQ


class DeepGLVQ(nn.Module):


    def __init__(self,n_dims_for_classif,
                    train_data,validation_data,test_data,lr = 1e-4,n_epochs = 20,batch_size = 256,n_prototypes_per_class = 8,gamma = 0):

        super().__init__()

        self.gamma = gamma
        
        self.n_dims_for_classif = n_dims_for_classif
        
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        self.n_prototypes_per_class = n_prototypes_per_class

        self.lr = lr

        self.n_epochs = n_epochs

        ### Defining Encoder

        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 194)
        self.fc3 = nn.Linear(194, self.n_dims_for_classif)

        self.bn1 = nn.BatchNorm1d(392)
        self.bn2 = nn.BatchNorm1d(194)

        ###

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.split_all_data_and_labels()


    def split_data_and_labels(self,data):

        all_data = []
        all_labels = []

        for data_label_pairing in data:

            data,label = data_label_pairing

            all_data.append(data)
            all_labels.append(label)

        all_data = torch.Tensor(np.array(all_data).squeeze(axis = 1))
        all_labels = torch.Tensor(all_labels)

        return all_data,all_labels
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.bn1(x)  
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc2(x)
        x = self.bn2(x)  
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc3(x)  

        return x


    #### Initializing the prototypes in the latent space
    def init_prototypes(self):
        
        classes = torch.unique(self.train_labels)
        n_classes = len(classes)
        
        train_latents = self.forward(self.train_data)
        
        P = torch.empty(n_classes * self.n_prototypes_per_class,self.n_dims_for_classif)
        P_idxs = torch.empty(n_classes*self.n_prototypes_per_class,)
        
        Py = classes.repeat_interleave(self.n_prototypes_per_class)
        
        data_indices = torch.arange(0,len(train_latents))

        for class_idx in range(n_classes):
        
            class_mask = self.train_labels == class_idx
        
            cur_prototype_indices = data_indices[class_mask][torch.randperm(torch.sum(class_mask))][:self.n_prototypes_per_class]
            cur_prototypes = train_latents[cur_prototype_indices]
        
            P[class_idx * self.n_prototypes_per_class : (class_idx + 1) * self.n_prototypes_per_class] = cur_prototypes
            P_idxs[class_idx * self.n_prototypes_per_class: (class_idx + 1) * self.n_prototypes_per_class] = cur_prototype_indices
        
        P_idxs = P_idxs.int()

        return P,Py,P_idxs
        

    def L_GLVQ(self,X, y, P, Py, gamma = 0, epsilon=1e-8):
        pairwise_distances = torch.cdist(X, P)  
    
        same_class_mask = (Py == y.unsqueeze(1))
        diff_class_mask = (Py != y.unsqueeze(1))
        
        d_plus = torch.where(same_class_mask, pairwise_distances, torch.full_like(pairwise_distances, np.inf))
        d_plus = torch.min(d_plus, dim=1).values  
    
        d_minus = torch.where(diff_class_mask, pairwise_distances, torch.full_like(pairwise_distances, np.inf))
        d_minus = torch.min(d_minus, dim=1).values  
    
        raw_loss = torch.sigmoid((d_plus - d_minus) / (d_plus + d_minus + epsilon))
        
        loss = torch.mean(torch.where(raw_loss > gamma, raw_loss - gamma, torch.zeros_like(raw_loss)))
    
        return loss



    def get_metrics(self, P, Py, on='validation'):
        train_latents = self.forward(self.train_data).detach()
        
        if on == 'validation':
            eval_latents = self.forward(self.validation_data).detach()
            eval_labels = self.validation_labels.detach()
        elif on == 'test':
            eval_latents = self.forward(self.test_data).detach()
            eval_labels = self.test_labels.detach()
        elif on == 'train':
            eval_latents = self.forward(self.train_data).detach()
            eval_labels = self.train_labels.detach()
    
        dists = torch.cdist(eval_latents, P)  
    
        pred_indices = torch.argmin(dists, dim=1)
        preds = Py[pred_indices] 
    
        acc = torch.sum(preds == eval_labels) / len(preds)

        loss = self.L_GLVQ(
            train_latents,self.train_labels,P,Py,gamma = self.gamma
        )
    
        return loss, acc

        
    
    def split_all_data_and_labels(self):     
        self.train_data,self.train_labels = self.split_data_and_labels(self.train_data)
        self.validation_data,self.validation_labels = self.split_data_and_labels(self.validation_data)
        self.test_data,self.test_labels = self.split_data_and_labels(self.test_data)
        

    def train_encoder(self):

        self.train()
        #The prototypes, their associated labels, and their associated indexes
        P,Py,P_idxs = self.init_prototypes()

        #The original data points that are associated with the prototypes in the latent space
        orig_data_points = self.train_data[P_idxs]

        for epoch in range(self.n_epochs):

            for batch_idx,(data,target) in tqdm(enumerate(self.train_loader),desc = f'Epoch {epoch + 1} / {self.n_epochs}'):
    
                self.optimizer.zero_grad()
                
                cur_batch_latents = self.forward(data)

                P_current = self.forward(orig_data_points)
                
                glvq_loss = self.L_GLVQ(
                    cur_batch_latents,target,P_current,Py,gamma = self.gamma
                )
    
                glvq_loss.backward()
                self.optimizer.step()

            val_loss,val_acc = self.get_metrics(P_current,Py,on = 'validation')
            train_loss,train_acc = self.get_metrics(P_current,Py,on = 'train')

            print(f'Epoch {epoch + 1} validation loss: {val_loss}')
            print(f'Epoch {epoch + 1} validation accuracy: {val_acc}\n')

            print(f'Epoch {epoch + 1} train loss: {train_loss}')
            print(f'Epoch {epoch + 1} train accuracy: {train_acc}\n')

        test_loss,test_acc = self.get_metrics(P_current,Py,on = 'test')

        print(f'Test loss is {test_loss} and test accuracy is {test_acc}')

    
                














            

            

    