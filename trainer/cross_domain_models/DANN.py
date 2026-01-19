import sys

sys.path.append("..")
from utils import *
from data.mydataset import create_dataset_full
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import copy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
from models.models import get_backbone_class, Model, Discriminator_DANN, ReverseLayerF

def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, src_train_dl, src_test_dl, tgt_train_dl, tgt_test_dl, src_id, tgt_id, run_id):

    print(f'From_source:{src_id}--->target:{tgt_id}...')

    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_{src_id}.pt')

    # pretrained source model
    source_model = Model(dataset_configs, backbone).to(device) 

    print('=' * 89)
 

    source_model.train()
    target_model = source_model

    
    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.CrossEntropyLoss()
    
    domain_classifier = Discriminator_DANN(dataset_configs).to(device)
    discriminator_optim = torch.optim.AdamW(domain_classifier.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_model.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    
    num_batches = max(len(src_train_dl), len(tgt_train_dl))
    
    best_score, best_rmse, best_risk = 0, 0, 1e5        
    for epoch in range(1, hparams['num_epochs'] + 1):
        if len(src_train_dl) > len(tgt_train_dl):
            joint_loader =enumerate(zip(src_train_dl, loop_iterable(tgt_train_dl)))
        else:
            joint_loader =enumerate(zip(loop_iterable(src_train_dl), tgt_train_dl))

        total_loss = 0
        start_time = time.time()
        
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            p = float(step + epoch * num_batches) / hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            # zero grad
            target_optim.zero_grad()
            discriminator_optim.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(device)
            domain_label_trg = torch.zeros(len(trg_x)).to(device)

            src_pred, src_feat = target_model(src_x) 
            _, trg_feat = target_model(trg_x)

            # Task classification  Loss
            src_cls_loss = criterion(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = domain_classifier(src_feat_reversed)
            src_domain_loss = dis_critierion(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = domain_classifier(trg_feat_reversed)
            trg_domain_loss = dis_critierion(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss
            
            loss = hparams["src_cls_loss_wt"] * src_cls_loss + hparams["domain_loss_wt"] * domain_loss
            loss.backward()
            target_optim.step()
            discriminator_optim.step()
                
            total_loss += loss.item()
                
        mean_loss = total_loss / (step+1)
        
        print(f'Epoch: {epoch:02}')
        print(f'target_loss:{mean_loss} ')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
            src_risk, _, _, _, _, _ = evaluate(target_model, src_train_dl, criterion, dataset_configs, device, score_flag=False)
            if best_risk > src_risk:
                best_rmse, best_score, best_risk = test_loss, test_score, src_risk
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            
            
    src_only_loss, src_only_score, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
    test_loss, test_score, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
       
    
    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    
    return best_risk, src_only_score, best_rmse, best_score
