from torch import nn
from multimodal import EarlyFusion

class SpeakerListenerSMEPI(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        # self.video_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
        # self.video_enc.load_state_dict(torch.load(f"{config.ckpt_root}/rgb_imagenet.pt"))

        self.audio_enc = HubertBase(config)
        self.MAX_VEC = torch.tensor(MAX_VEC).cuda()
        self.MAX_VEC = torch.maximum(self.MAX_VEC, torch.tensor(1).cuda())
        self.MIN_VEC = torch.tensor(MIN_VEC).cuda()
        
        # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.text_enc = RobertaModel.from_pretrained("roberta-base")

        self.mlp1_spk = nn.Linear(dim_audio, 64)
        self.mlp1_list = nn.Linear(dim_openface, 64)

        self.fusion_mlp = nn.Linear(2 * 64, 64)

        self.xatt = nn.MultiheadAttention(...)
        self.mlp_final = nn.Linear(64, 1)

    def fusion_vec(x):
        
    
    def forward(self, x_spk, x_lis):
        x_spk = self.audio_enc.extract_features(x_spk).view(B, -1)
        x_spk = self.mlp1_spk(x_spk)
        x_lis = self.mlp1_list(x_lis)

        res = self.fusion_mlp(x_lis, x_spk)
        xatt = self.xatt(x_lis, x_spk)
        return self.mlp_final(res + xatt)





# Process facial features with MLP to get intermediate representation
        self.facial_mlp = nn.Sequential(
            nn.Linear(len(self.filteredcolumns), config.hidden_size),
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()),
            nn.Dropout(config.dropout)
        )

        # Process audio features with MLP to get intermediate representation
        self.audio_mlp = nn.Sequential(
            nn.Linear(768, config.hidden_size),  # 768 is the audio feature size
            (nn.Tanh() if self.config.activation_fn == "tanh" else nn.LeakyReLU()),
            nn.Dropout(config.dropout)
        )