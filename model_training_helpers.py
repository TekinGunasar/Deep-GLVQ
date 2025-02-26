import torch.optim as optim
import numpy as np

from tqdm import tqdm



def get_metrics(model, loader, loss_fn):
    loss_sum_across_batches = 0
    acc_sum_across_batches = 0
    total_batches = len(loader)

    for batch_idx, (data, target) in enumerate(loader):

        output = model(data)
        loss = loss_fn(output, target)

        loss_sum_across_batches += loss.item()  

        predicted_labels = output.argmax(dim=1).cpu().numpy()
        actual_labels = target.cpu().numpy()

        batch_accuracy = np.mean(predicted_labels == actual_labels) 
        acc_sum_across_batches += batch_accuracy

    avg_loss = loss_sum_across_batches / total_batches
    avg_acc = acc_sum_across_batches / total_batches

    return avg_loss, avg_acc


def train_model(model,loss_fn,n_epochs,train_loader,validation_loader):

    optimizer = optim.Adam(model.parameters(),lr = 1e-4)

    model.train()

    for epoch in range(n_epochs):
    
        for batch_idx,(data,target) in tqdm(enumerate(train_loader),desc = f'Epoch {epoch + 1} / {n_epochs}'):
    
            optimizer.zero_grad()
        
            output = model(data)
            
            loss = loss_fn(output,target)
            
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

        train_loss,train_acc = get_metrics(
            model,train_loader,loss_fn
        )

        val_loss,val_acc = get_metrics(
            model,validation_loader,loss_fn
        )

        print(f'Epoch {epoch + 1} train loss: {train_loss}')
        print(f'Epoch {epoch + 1} train accuracy: {train_acc}\n')

        print(f'Epoch {epoch + 1} validation loss: {val_loss}')
        print(f'Epoch {epoch + 1} validation accuracy: {val_acc}\n')
              





    