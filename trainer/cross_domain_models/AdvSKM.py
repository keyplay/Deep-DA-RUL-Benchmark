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
from models.models import get_backbone_class, Model, AdvSKM_Disc

def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, src_train_dl, src_test_dl, tgt_train_dl, tgt_test_dl, src_id, tgt_id, run_id):

    print(f'From_source:{src_id}--->target:{tgt_id}...')

    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_{src_id}.pt')

    # pretrained source model
    source_model = Model(dataset_configs, backbone).to(device) 
    AdvSKM_embedder = AdvSKM_Disc(dataset_configs).to(device)
    print('=' * 89)
 
    if hparams['pretrain']:
        source_model.load_state_dict(checkpoint['state_dict'])
        source_model.eval()
        set_requires_grad(source_model, requires_grad=False)
        
        # initialize target model
        target_model = Model(dataset_configs, backbone).to(device)
        target_model.load_state_dict(checkpoint['state_dict'])
        target_encoder = target_model.feature_extractor
        target_encoder.train()
        set_requires_grad(target_encoder, requires_grad=True)
    else:
        source_model.train()
        target_model = source_model

    
    # criterion
    criterion = RMSELoss()
    mmd_criterion = MMDLoss()
    # optimizer
    target_optim = torch.optim.AdamW(target_model.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))   
    optimizer_disc = torch.optim.Adam(AdvSKM_embedder.parameters(), lr=hparams["learning_rate"],weight_decay=hparams["weight_decay"])
    
    best_score, best_rmse, best_risk = 0, 0, 1e5        
    for epoch in range(1, hparams['num_epochs'] + 1):
        if len(src_train_dl) > len(tgt_train_dl):
            joint_loader =enumerate(zip(src_train_dl, loop_iterable(tgt_train_dl)))
        else:
            joint_loader =enumerate(zip(loop_iterable(src_train_dl), tgt_train_dl))

        total_loss = 0
        start_time = time.time()
        for step, ((source_x, source_y), (target_x, _)) in joint_loader:
            #print(source_y) 
            target_optim.zero_grad()

            source_x, target_x, source_y = source_x.to(device), target_x.to(device), source_y.to(device) 
            src_pred, src_feat = source_model(source_x)              
            _, trg_feat = target_model(target_x)

            source_embedding_disc = AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - mmd_criterion(source_embedding_disc, target_embedding_disc)

            # update discriminator
            optimizer_disc.zero_grad()
            mmd_loss.backward()
            optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = criterion(src_pred.squeeze(), source_y)

            # domain loss.
            source_embedding_disc = AdvSKM_embedder(src_feat)
            target_embedding_disc = AdvSKM_embedder(trg_feat)

            mmd_loss_adv = mmd_criterion(source_embedding_disc, target_embedding_disc)

            # calculate the total loss
            loss = hparams["domain_loss_wt"] * mmd_loss_adv + hparams["src_cls_loss_wt"] * src_cls_loss
            
            loss.backward()
            target_optim.step()
            total_loss += loss.item()
                
        mean_loss = total_loss / (step+1)
        
        print(f'Epoch: {epoch:02}')
        print(f'target_loss:{mean_loss} ')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
            src_risk, _, _, _, _, _ = evaluate(target_model, src_train_dl, criterion, dataset_configs, device)
            if best_risk > src_risk:
                best_rmse, best_score, best_risk = test_loss, test_score, src_risk
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            
            
    src_only_loss, src_only_score, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
    test_loss, test_score, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
       
    
    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    
    return best_risk, src_only_score, best_rmse, best_score
