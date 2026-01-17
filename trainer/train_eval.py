import torch
from utils import scoring_func

def train(model, train_dl, optimizer, criterion,config,device):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    for inputs, labels in train_dl:
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred, feat = model(src)
        #loss and score
        rul_loss = criterion(pred.squeeze(), labels)
        
        #denormalization
        pred  = pred * config.max_rul
        labels = labels * config.max_rul
        score = scoring_func(pred.squeeze() - labels)

        rul_loss.backward()
        if (type(model.feature_extractor).__name__=='LSTM'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score
    return epoch_loss / len(train_dl), epoch_score, pred, labels



def evaluate(model, test_dl, criterion, config, device, denorm_flag=True, score_flag=True):
    model.eval()
    total_feas=[];total_labels=[]
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            
            labels = labels.to(device)

            pred, feat = model(src)
            # denormalize predictions
            if denorm_flag:
                pred = pred * config.max_rul
                if labels.max() <= 1:
                    labels = labels * config.max_rul
            rul_loss = criterion(pred.squeeze(), labels)
            
            if score_flag: score = scoring_func(pred.squeeze() - labels)
            else: score = 0
            
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()

    model.train()
    return epoch_loss / len(test_dl), epoch_score, torch.cat(total_feas), torch.cat(total_labels),predicted_rul,true_labels

