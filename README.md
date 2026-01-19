# Deep Domain Adaptation for Turbofan Engine Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends [[Paper](https://arxiv.org/abs/2510.03604)]
#### *by: Yucheng Wang, Mohamed Ragab, Yubo Hou, Zhenghua Chen, Min Wu, and Xiaoli Li*

## Requirmenets:
- Python3.x
- Pytorch==1.12.1
- Numpy
- Sklearn
- Pandas

## Train the model
To pre train model:

```
python pretrain_main.py 
```
To train model for domain adaptation:

```
python main_cross_domains.py --da_method [method] --dataset [dataset]    
```
