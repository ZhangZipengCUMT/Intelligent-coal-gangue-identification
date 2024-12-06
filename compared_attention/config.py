import yaml
from transformers.configuration_utils import PretrainedConfig
from RetNet_v2.retnet.configuration_retnet import RetNetConfig


RetNet_param = RetNetConfig(vocab_size=342,
                            num_heads=8,
                            chunk_size=32,
                            num_layers=3,
                            hidden_size=32,
                            qk_dim=128,
                            v_dim=256,
                            ffn_proj_size=256,
                            pad_token_id=341,
                            eos_token_id=341,
                            use_default_gamma=True)

def load_config_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = AffnetConfig.from_dict(config)
    return config


class AffnetConfig(PretrainedConfig):
    model_type = "Affnet"
    def __init__(self,
                 afno2d_sparsity_threshold: float = 0.01,
                 afno2d_name: str = "relu",
                 afno2d_neg_slope: float = 0.1,
                 afno2d_activation_inplace: bool = False,

                 invres_norm_name: str = "batch_norm",
                 invres_activation_name: str = "prelu",
                 invres_neg_slope: float = 0.1,
                 invres_activation_inplace: bool = False,

                 norm_groups: int = 1,
                 norm_name: str = "batch_norm",
                 norm_momentum: float = 0.1,

                 enable_coreml_compatible_module: bool = False,
                 use_jit_model: bool = False,

                 dim: int = 1,
                 hidden_size: int = 1,
                 num_blocks: int = 1,
                 double_skip: int = True,
                 attn_norm_layer: int = "batch_norm",

                 **kwargs):
        self.AFNO2D_sparsity_threshold = afno2d_sparsity_threshold
        self.AFNO2D_name = afno2d_name
        self.AFNO2D_neg_slope = afno2d_neg_slope
        self.AFNO2D_activation_inplace = afno2d_activation_inplace

        self.Norm_name = norm_name
        self.Norm_groups = norm_groups
        self.Norm_momentum = norm_momentum

        self.Invres_norm_name = invres_norm_name
        self.Invres_activation_name = invres_activation_name
        self.Invres_neg_slope = invres_neg_slope
        self.Invres_activation_inplace = invres_activation_inplace

        self.Enable_coreml_compatible_module = enable_coreml_compatible_module
        self.Use_jit_model = use_jit_model

        self.dim = dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.double_skip = double_skip
        self.attn_norm_layer = attn_norm_layer

        super().__init__(**kwargs)