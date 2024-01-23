
# code from 
# https://github.com/josehoras/Knowledge-Distillation

from tqdm import tqdm
import os
from mnist_loader import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import linear_net, small_linear_net
from utils import get_device
from data import get_dataloaders
from evaluate import evaluate
device = get_device()
batch_size = 10 
load = False 
temperatures = [1,2,3,4,5,7,5,10,15,20]
epochs = 10
lr = 1e-3 

checkpoints_path = "checkpoints"
if not os.path.exists(checkpoints_path): 
    os.mkdir(checkpoints_path)

    
teacher_checkpoint_path =os.path.join(checkpoints_path, "teacher")
if not os.path.exists(teacher_checkpoint_path): 
    print (f'ERROR : teacher model does not exist')

teacher_model = linear_net().to(device)

kd_distill_checkpoint_path = os.path.join(checkpoints_path, "distill")
if not os.path.exists(kd_distill_checkpoint_path): 
    os.mkdir(kd_distill_checkpoint_path)
    
teacher_checkpoint = torch.load( os.path.join(teacher_checkpoint_path, "model"))
teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
teacher_model.eval()



train_loader, val_loader, test_loader = get_dataloaders(batch_size)
softmax_op = nn.Softmax(dim=1)

# loss 
mseloss_fn = nn.MSELoss()


def my_loss(scores, targets, T=5): 
    return mseloss_fn(softmax_op(scores/T), softmax_op(targets/T))
    


def train(): 

    models = {}

    for temp in temperatures: 
        title = f'T={temp}'
        student_model = small_linear_net().to(device)
        
        optimizer= Adam(student_model.parameters(), lr=lr)
        
        val_acc = []
        train_acc = []
        train_loss = [-np.log(1.0/10)] # loss at iteration 0 
        it_per_epoch = len(train_loader)
        it=0
        
        if load == True: 
            checkpoint = torch.load(kd_distill_checkpoint_path + "model")
            student_model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint['optimizer_state_dict']['params_group'][0]['lr'] = 5e-4
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print (checkpoint['optimizer_state_dict']['params_group'])
            print (checkpoint.keys())
            val_acc = checkpoint['val_acc']
            train_acc = checkpoint['train_acc']
            train_loss = checkpoint['loss_hist']
            it = checkpoint['iterations']
            
        for epoch in range(epochs): 
            for features, labels in tqdm(train_loader): 
                scores = student_model(features)
                targets = teacher_model(features)
                
                loss = my_loss(scores, targets, T=temp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                
                if it % 500 ==0 : 
                    train_acc.append(evaluate(student_model, train_loader, max_ex=100))
                    val_acc.append(evaluate(student_model, val_loader))
                    print (f'Iterations : {it}, val acc : {val_acc[-1]}')
                it+=1 
                
            # perform on last iteration 
            train_acc.append(evaluate(student_model, train_loader, max_ex=100))
            val_acc.append(evaluate(student_model, val_loader))
            print (f'val acc after {epoch}/{epochs}: {val_acc[-1]}')
            
            models[title]= {'model': student_model,
                            'model_state_dict': student_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr': lr, 
                            'T': temp,
                            'loss_hist': train_loss, 
                            'train_acc': train_acc, 
                            'val_acc': val_acc,
                            'iteration': iter
                            }

 
    # for key in models.keys():
    #     # print (f'lr : {models[key]['lr']}, val_acc : {models[key]['val_acc']}')
    #     print (f'lr : {models[key]}, val_acc : {models[key]}')
        

    val_accs = [models[key]['val_acc'][-1] for key in models.keys()]
    xs = [models[key]['T'] for key in models.keys()]
    keys = [key for key in models.keys()]

    print (f'val_acc : {val_accs}')
    print (f'keys : {keys}')
    print (f'Best model val acc : {np.argmax(val_accs)}')
    fig = plt.figure(figsize=(8,4), dpi=100)
    plt.scatter(xs, val_accs)
    plt.title(f'{epochs} Epochs')
    plt.ylabel('val acc')
    plt.xlabel('T')
    fig.savefig(f'summary_{epochs}_epochs.png')

    best_key=keys[np.argmax(val_accs)]
    print (f'best key : {best_key}')
    best_model = models[best_key]['model']
    best_model.eval()

    torch.save(models[best_key], os.path.join(kd_distill_checkpoint_path + 'model'))


def eval_for_all(model, dataloaders): 
    train_acc = evaluate(model, dataloaders['train'])
    val_acc = evaluate(model, dataloaders['val'])
    test_acc = evaluate(model, dataloaders['test'])
    
    return train_acc, val_acc, test_acc

def load_model(name='teacher', device='cpu'): 
    checkpoint_path =os.path.join(checkpoints_path, name)
    if not os.path.exists(checkpoint_path): 
        print (f'ERROR : {name} model checkpoint does not exist')

    if name == 'teacher': 
        model = linear_net().to(device)
    elif name == 'student' or 'distill': 
        model = small_linear_net().to(device)
    else : 
        print (f'ERROR : {name} model is not defined')
        return
    checkpoint = torch.load( os.path.join(checkpoint_path , 'model'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    
    return model 

     

def compare_models():
    # load models 
    teacher_model = load_model(name = "teacher", device=get_device())
    student_model = load_model(name = "student", device=get_device())
    distill_model = load_model(name='distill', device=get_device())
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    dataloaders = {}
    dataloaders['train'],dataloaders['val'], dataloaders['test']  = train_loader, val_loader, test_loader
    
    print (f'Teacher model : ')
    train_acc, val_acc, test_acc = eval_for_all(teacher_model, dataloaders)
    print(f'train acc : {train_acc}, val_acc : {val_acc}, test_acc : {test_acc}')
    

    print (f'student model : ')
    train_acc, val_acc, test_acc = eval_for_all(student_model, dataloaders)
    print(f'train acc : {train_acc}, val_acc : {val_acc}, test_acc : {test_acc}')


    print (f'Distilled model : ')
    train_acc, val_acc, test_acc = eval_for_all(distill_model, dataloaders)
    print(f'train acc : {train_acc}, val_acc : {val_acc}, test_acc : {test_acc}')


if __name__ == "__main__": 
    # train()
    compare_models()
    
    