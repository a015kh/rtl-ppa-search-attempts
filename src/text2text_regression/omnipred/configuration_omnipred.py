from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class OmniPredConfig(PretrainedConfig):
    model_type = "omnipred"

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_input_len: int = 2048,
        encoder_vocab_size: int = 32128,
        decoder_vocab_size: int = 37,
        max_encoder_len: int = 2048,
        max_decoder_len: int = 2048,
        num_digits: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_input_len = max_input_len
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.num_digits = num_digits
