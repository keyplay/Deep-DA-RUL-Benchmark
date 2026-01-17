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
from models.models import Model, Discriminator, NCE_model
import wandb


def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, src_train_dl, src_test_dl, tgt_train_dl, tgt_test_dl, src_id, tgt_id, run_id):

    print(f'From_source:{src_id}--->target:{tgt_id}...')

    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_{src_id}.pt')
    # pretrained source model
    source_model = Model(dataset_configs, backbone).to(device) 

    print('=' * 89)
 
    source_model.load_state_dict(checkpoint['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    source_encoder = source_model.feature_extractor

    # initialize target model
    target_model = Model(dataset_configs, backbone).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    target_encoder = target_model.feature_extractor
    target_encoder.train()
    set_requires_grad(target_encoder, requires_grad=True)

    # discriminator network
    domain_classifier = Discriminator(dataset_configs).to(device)
    comput_nce= NCE_model(dataset_configs, device).to(device)
    
    # criterion
    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    discriminator_optim = torch.optim.AdamW(domain_classifier.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))
    nce_optim = torch.optim.AdamW(comput_nce.parameters(), lr=hparams['nce_learning_rate'], betas=(0.5, 0.9))

    best_score, best_rmse, best_risk = 0, 0, 1e5
    for epoch in range(1, hparams['num_epochs'] + 1):
        if len(src_train_dl) > len(tgt_train_dl):
            joint_loader =enumerate(zip(src_train_dl, loop_iterable(tgt_train_dl)))
        else:
            joint_loader =enumerate(zip(loop_iterable(src_train_dl), tgt_train_dl))
            
        total_loss = 0
        total_accuracy = 0
        alpha = hparams['nce_loss_wt'] 
        target_losses, nce = 0, 0

        for step, ((source_x, _), (target_x, _)) in joint_loader:
            # Train discriminator
            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(domain_classifier, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            _, source_features = source_model(source_x)
            _, target_features = target_model(target_x)

            discriminator_x = torch.cat([source_features.view(source_features.shape[0], -1), target_features.view(target_features.shape[0], -1)])
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                         torch.zeros(target_x.shape[0], device=device)])
            preds = domain_classifier(discriminator_x).squeeze()
            loss = dis_critierion(preds, discriminator_y)
            discriminator_optim.zero_grad()
            loss.backward()
            discriminator_optim.step()
            total_loss += loss.item()
            total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
            
            # Train Feature Extractor
            if step % hparams['k_disc'] == 0:
                set_requires_grad(target_encoder, requires_grad=True)
                set_requires_grad(domain_classifier, requires_grad=False)

                target_optim.zero_grad()
                nce_optim.zero_grad()
            
                _, target_features = target_model(target_x)
                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)
                preds = domain_classifier(target_features.view(target_features.shape[0], -1)).squeeze()
                target_loss = dis_critierion(preds, discriminator_y)
                # Negaative Contrastive Estimtion Loss
                nce_loss= comput_nce(target_features.view(target_features.shape[0], -1), target_x)
                #total loss
                loss = target_loss + alpha*nce_loss
                loss.backward()
                target_optim.step()
                nce_optim.step()
                target_losses += target_loss.item()
                nce += nce_loss.item()
        
        mean_loss = total_loss / (step+1)
        mean_accuracy = total_accuracy /(step+1)
        mean_nce = nce / (step+1) * hparams['k_disc']
        mean_tgt_loss = target_losses / (step+1) * hparams['k_disc']

        # tensorboard logging
        
        print(f'Epoch: {epoch + 1:02}')
        print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy{mean_accuracy}')
        print(f'target_loss:{mean_tgt_loss}  \t NCE_loss{mean_nce}')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs, device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs, device)
            src_risk, _, _, _, _, _ = evaluate(target_model, src_train_dl, criterion, dataset_configs, device)
            if best_risk > src_risk:
                best_rmse, best_score, best_risk = test_loss, test_score, src_risk
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            
            
    src_only_loss, src_only_score, _, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
    test_loss, test_score, _, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
    
    
    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
   
    return best_risk, src_only_score, best_rmse, best_score
