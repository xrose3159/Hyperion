class ModelParser():
    def __init__(self):
        super().__init__()
        self.device = 0
        self.readout = 'max'                   
        self.enable_prot = True                       
        self.num_prototypes_per_class = 4           
        self.prot_dim = 32
        self.single_target = True
        self.mlp_out_dim = 0           

model_args = ModelParser()
