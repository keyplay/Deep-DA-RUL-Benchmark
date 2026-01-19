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

class NCMAPSS_F():
    def __init__(self):
        super(NCMAPSS_F, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "SASA": {
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "DDC": {
                "learning_rate": 0.001,
                "mmd_wt": 3.7991920933520342,
                "src_cls_loss_wt": 6.286301875125623,
                "domain_loss_wt": 6.36,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "domain_loss_wt": 3.2750474868706925,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.335109786953256,
                "weight_decay": 0.0001
            },
            "DANN": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            },
            "CADA": {
                "nce_loss_wt": 1,
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.20721518968644,
                "domain_loss_wt": 1.9012145515129044,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.67861021290254,
                "vat_loss_wt": 7.7102843136045855,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "learning_rate": 0.001,
                "mmd_wt": 2.0872340713147786,
                "src_cls_loss_wt": 1.8744909939900247,
                "domain_loss_wt": 1.59,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "cond_ent_wt": 1.383002023133561,
                "coral_wt": 8.36810764913737,
                "learning_rate": 0.001,
                "mmd_wt": 3.964042918489996,
                "src_cls_loss_wt": 6.794522068759213,
                "weight_decay": 0.0001
            },
            "Deep_Coral": {
                "coral_wt": 4.23035475456397,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1013209750429822,
                "weight_decay": 0.0001
            },
            "CDAN": {
                "cond_ent_wt": 1.2920143348777362,
                "domain_loss_wt": 9.545761950873414,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 9.430292987535724,
                "weight_decay": 0.0001
            },
            "AdvSKM": {
                "domain_loss_wt": 1.338788378230754,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.468525942065072,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "hommd_wt": 2.8305712579412683,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1282520874653523,
                "domain_loss_wt": 9.13,
                "weight_decay": 0.0001
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.9, 'temporal_shift': 14,
                       'src_cls_weight': 0.78, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05},
            'MCD': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43},

        }




