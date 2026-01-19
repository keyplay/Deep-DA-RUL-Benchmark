## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CMAPSS():
    def __init__(self):
        super(CMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0, #0.5
            'pretrain': False,
            "domain_loss_wt": 1,

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "DDC": {
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                #"domain_loss_wt": 100,
                "weight_decay": 0.0001
            },
            "ADARUL": {
                "learning_rate": 5e-5,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "DANN": {
                #"domain_loss_wt": 2.943729820531079,
                "src_cls_loss_wt": 1,
                "learning_rate": 5e-3,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "CADA": {
                "nce_loss_wt": 0.2,
                "learning_rate": 5e-5,
                "pretrain_learning_rate": 0.001,
                "nce_learning_rate": 0.01,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001,
                "k_disc": 1
            },
            "Deep_Coral": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "SASA": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "AdvSKM": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "HoMM": {
                #"domain_loss_wt": 1.338788378230754,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "consDANN": {
                #"domain_loss_wt": 1.338788378230754,
                "cons_loss_wt": 1,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
        }


class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'pretrain': False,
            "domain_loss_wt": 1,

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "DDC": {
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                #"domain_loss_wt": 100,
                "weight_decay": 0.0001
            },
            "ADARUL": {
                "learning_rate": 5e-4,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "DANN": {
                #"domain_loss_wt": 2.943729820531079,
                "src_cls_loss_wt": 1,
                "learning_rate": 5e-3,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "CADA": {
                "nce_loss_wt": 0.2,
                "learning_rate": 5e-4,
                "pretrain_learning_rate": 0.001,
                "nce_learning_rate": 0.01,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001,
                "k_disc": 1
            },
            "Deep_Coral": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "SASA": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "weight_decay": 0.0001,
                "k_disc": 1,
            },
            "AdvSKM": {
                #"domain_loss_wt": 1e-3,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "HoMM": {
                #"domain_loss_wt": 1.338788378230754,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "consDANN": {
                #"domain_loss_wt": 1.338788378230754,
                "cons_loss_wt": 1,
                "learning_rate": 5e-3,
                "src_cls_loss_wt": 1,
                "weight_decay": 0.0001
            },
        }




