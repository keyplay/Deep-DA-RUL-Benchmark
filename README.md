# Deep Domain Adaptation for Turbofan Engine Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends [[Paper](https://arxiv.org/abs/2510.03604)]
#### *by: Yucheng Wang, Mohamed Ragab, Yubo Hou, Zhenghua Chen, Min Wu, and Xiaoli Li*

## Supported Method
- Source Only (without domain adaptation)
- [DDC](https://arxiv.org/abs/1412.3474)
- [Deep Coral](https://arxiv.org/abs/1612.01939)
- [DANN](https://arxiv.org/abs/1505.07818)
- [ADARUL](https://ieeexplore.ieee.org/document/9187053)
- [CADA](https://ieeexplore.ieee.org/document/9234721)
- [ConsDANN](https://ieeexplore.ieee.org/document/9741812)
- [HoMM](https://arxiv.org/abs/1912.11976)
- [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf)

## Supported Dataset
- [C-MAPSS](https://ieeexplore.ieee.org/document/4711414)
- [N-CMAPSS](https://www.mdpi.com/2306-5729/6/1/5)

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
