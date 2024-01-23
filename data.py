import torch 
from torch.utils.data import Dataset, DataLoader 
from utils import get_device
from mnist_loader import my_load_data_wrapper


    
class CustomDataset(Dataset): 
    
    def __init__(self, features, labels): 
        self.features = features 
        self.labels = labels 
        self.device = get_device()
        
    def __getitem__(self, idx): 
        
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        return (feature.to(self.device), label.to(self.device))

    def __len__(self): 
        return len(self.labels) 
    
    
def get_dataloaders(batch_size): 
    # create dataloader 
    X_train, y_train, X_val, y_val, X_test, y_test = my_load_data_wrapper()
    print (f'training data')
    print (f'=============')
    print (f'X_train : {X_train.shape}, X_val : {X_val.shape}, X_test : {X_test.shape}')

    train_data = CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_data = CustomDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_data = CustomDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


    

