
import torch 

def evaluate(model, dataset, max_ex=0): 
    model.eval()
    acc = 0 
    
    
    for i, (features, labels) in enumerate(dataset): 
        batch_size = len(features)
        scores = model(features)
        preds = torch.argmax(scores, dim=1)
        acc += torch.sum(torch.eq(preds, labels)).item()
        if max_ex !=0 and i >= max_ex:
            break           
    
    model.train()
    
    return ((acc * 100) / ((i+1) * batch_size))