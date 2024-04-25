import torch
import numpy as np 
device = torch.device('cuda')
import pandas as pd
from models.my_models import *
from models.models_config import get_model_config
#Different Domain Adaptation  approaches
import importlib
import random
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from data.mydataset import data_generator,create_dataset_full
from models.models import get_backbone_class, Model

print(torch.cuda.current_device())

dataset = 'CMAPSS'
num_runs=1



if __name__ == '__main__':
  select_method= 'Deep_Coral'
  backbone = 'CNN'
  method = importlib.import_module(f'trainer.cross_domain_models.{select_method}')
  
  data_path= "./data/datasets/"+dataset
  
  dataset_class = get_dataset_class(dataset)
  hparams_class = get_hparams_class(dataset)
  dataset_configs = dataset_class()
  hparams_class = hparams_class()
  hparams = {**hparams_class.alg_hparams[select_method], **hparams_class.train_params}
 
  df=pd.DataFrame();res = [];full_res = []
  print('=' * 89)
  print (f'Domain Adaptation using: {select_method}')
  print('=' * 89)
  
  for src_id, tgt_id in dataset_configs.scenarios:
      total_loss = []
      total_score = []

      seed = 42
      for run_id in range(num_runs):
          seed += 1
          torch.manual_seed(seed)
          np.random.seed(seed)
          random.seed(seed)
          
          src_train_dl = data_generator(data_path, src_id, dataset_configs, hparams, "train")
          src_test_dl = data_generator(data_path, src_id, dataset_configs, hparams, "test")
  
          tgt_train_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "train")
          tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")
          
          src_only_loss, src_only_score, test_loss, test_score = method.cross_domain_train(device,dataset,dataset_configs, hparams, backbone,src_train_dl, src_test_dl, tgt_train_dl, tgt_test_dl,src_id,tgt_id,run_id)
          total_loss.append(test_loss)
          total_score.append(test_score)
      loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
      score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
      full_res.append((f'run_id:{run_id}',f'{src_id}-->{tgt_id}', f'{src_only_loss:2.4f}' ,f'{loss_mean:2.4f}',f'{loss_std:2.4f}',f'{src_only_score:2.4f}',f'{score_mean:2.4f}',f'{score_std:2.4f}'))
              
  df= df.append(pd.Series((f'{select_method}')), ignore_index=True)
  df= df.append(pd.Series(("run_id", 'scenario','src_only_loss', 'mean_loss','std_loss', 'src_only_score', f'mean_score',f'std_score')), ignore_index=True)
  df = df.append(pd.DataFrame(full_res), ignore_index=True)
  print('=' * 89)
  print (f'Results using: {select_method}')
  print('=' * 89)
  print(df.to_string())
  df.to_csv(f'./results/results_{dataset}_{select_method}_{backbone}.csv')
