def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CMAPSS():
    def __init__(self):
        super(CMAPSS, self)
        self.scenarios = [("FD001", "FD002"), ("FD001", "FD003"), ("FD001", "FD004"), ("FD002", "FD001"), ("FD002", "FD003"), ("FD002", "FD004"),("FD003", "FD001"), ("FD003", "FD002"), ("FD003", "FD004"), ("FD004", "FD001"),("FD004", "FD002"), ("FD004", "FD003")]
        self.sequence_len = 30
        self.max_rul = 130
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.permute = False # True for lstm

        # model configs
        self.input_channels = 14
        self.kernel_size = 4
        self.stride = 2
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 32
        self.final_out_channels = 32
        self.features_len = 64 # 150 #128 #64 for CNN, 32 for SASA

        # TCN features
        self.tcn_layers = [32, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 32
        self.lstm_n_layers = 5
        self.lstm_bid = True

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self)
        self.scenarios = [("DS01", "DS02"), ("DS01", "DS03"), ("DS02", "DS01"), ("DS02", "DS03"), ("DS03", "DS01"), ("DS03", "DS02")]
        self.sequence_len = 50
        self.max_rul = 88
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.permute = False

        # model configs
        self.input_channels = 20
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.1
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 20
        self.final_out_channels = 20
        self.features_len = 160 #160 for cnn, 128 for lstm, 64 for tcn, 

        # TCN features
        self.tcn_layers = [32, 64] #[75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.1

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class NCMAPSS_F():
    def __init__(self):
        super(NCMAPSS_F, self)
        self.scenarios = [("flight1", "flight2"), ("flight1", "flight3"), ("flight2", "flight1"), ("flight2", "flight3"), ("flight3", "flight1"), ("flight3", "flight2")]
        self.sequence_len = 50
        self.max_rul = 88
        self.shuffle = True
        self.drop_last = True
        self.normalize = False

        # model configs
        self.input_channels = 20
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 20
        self.final_out_channels = 20
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


        
