from tqdm import tqdm
from mnist_loader import my_load_data_wrapper
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import linear_net, small_linear_net
from data import CustomDataset
from evaluate import evaluate
from utils import get_device
import argparse 



def train(model_type="teacher"): 
    
    
    base_dir = "checkpoints"
    
    if model_type == "teacher":
        model_checkpoint_dir = os.path.join(base_dir, "teacher")
        print (f'================')
        print(f'training teacher')
        print (f'================')
    else : 
        model_checkpoint_dir = os.path.join(base_dir, "student")
        print (f'================')
        print(f'training student')
        print (f'================')
    

    device = get_device()
    if not os.path.exists(base_dir): 
        os.mkdir(base_dir)

    if not os.path.exists(model_checkpoint_dir): 
        os.mkdir(model_checkpoint_dir)
    
    batch_size = 100
    lr = 1e-3 

    # loss function 
    loss_fn = nn.CrossEntropyLoss()

    # hyper parameters 
    lrs = [5e-4]
    dropouts = [0.4]
    epochs = 10

    models = {}
    old_val_acc = 0 

    train_loader, val_loader, test_loader = get_dataloaders(batch_size)


    for lr in lrs : 
        for dropout in dropouts : 
            title = f'dropout p={dropout}' 
            print (title)
            
            if model_type == "teacher": 
                net = linear_net(dropout=dropout).to(device)
            else :
                net = small_linear_net(dropout=dropout).to(device)
            
            optimizer = Adam(net.parameters(), lr=lr)
            optimizer.zero_grad()
            
            val_acc = [] 
            train_acc = []
            train_loss = [-np.log(1.0/10)] # loss at iternation zero
            
            iter_per_epoch = len(train_loader)
            # print (iter_per_epoch)

            it = 0 
            for epoch in range(epochs) : 
                for features, labels in tqdm(train_loader): 
                    scores = net(features)
                    loss = loss_fn(scores, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    if it % 100 == 0 : 
                        train_acc.append(evaluate(net, train_loader, max_ex=100))
                        val_acc.append(evaluate(net, val_loader))
                    
                    it+=1
                t_acc = evaluate(net, train_loader, max_ex=100)
                v_acc = evaluate(net, val_loader)
                
                print(f'epoch : {epoch}/{epochs}, train acc : {t_acc}, val_acc : {v_acc}')
            train_acc.append(evaluate(net, train_loader, max_ex=100))
            val_acc.append(evaluate(net, val_loader))
                    
            models[title] = {   'model': net,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss_hist': train_loss,
                                'lr':lr,
                                'p':dropout,
                                'val_acc': val_acc[-1]
                            }
                
                              
                              
    # for key in models.keys(): 
    #     print (f'lr : {models[key][lr]}, dropout : {models[key]['val_acc']}'))     



    val_accs = [models[key]['val_acc'] for key in models.keys()]
    xs = [models[key]['p'] for key in models.keys()]
    keys = [key for key in models.keys()]

    best_key = keys[np.argmax(val_accs)]
    print(best_key)
    best_model = models[best_key]['model']

    train_acc = evaluate(net, train_loader)
    val_acc = evaluate(net, val_loader)
    test_acc = evaluate(net, test_loader)

    print (f'train acc : {train_acc}, val_acc = {val_acc}, test_acc = {test_acc}')

    torch.save({'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss},
                os.path.join(model_checkpoint_dir, "model"))


def main(): 
    parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='train model with KD',
                    epilog='Text at the bottom of help')
    parser.add_argument('-model-type', default='teacher', help='model type teacher, student]')
    args = parser.parse_args()
    
    for model_type in ["teacher", "student"]: 
        train(model_type)
    
if __name__ == "__main__": 
    main()    
        


ls 

