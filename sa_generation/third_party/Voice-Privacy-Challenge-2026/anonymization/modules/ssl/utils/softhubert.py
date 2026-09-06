import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import fairseq

class SoftPredictor(nn.Module):
    def __init__(self, ssl_model):
        super().__init__()
        self.ssl_model = ssl_model
        self.ssl_features = 768
        self.km_class = 200
        self.W = nn.Parameter(torch.randn(self.km_class, self.ssl_features), requires_grad=True)

    def forward(self, wav):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        x = F.normalize(x)
        W = F.normalize(self.W)
        return F.linear(x, W)

class ContentExtractor:
    """HuBERT-based content feature extractor"""
    
    def __init__(self, hubert_model_path: str, soft_model_path: str, device: str = "cpu"):
        self.device = device
        
        # Load HuBERT model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [hubert_model_path], strict=False, arg_overrides={"weights_only": False}
            )
        
        feat_model = feat_model[0]
        feat_model.remove_pretraining_modules()
        
        # Load soft predictor
        self.soft_model = SoftPredictor(feat_model)
        self.soft_model.load_state_dict(torch.load(soft_model_path, map_location=device))
        self.soft_model.to(device).eval()
        
        # Freeze parameters
        for p in self.soft_model.parameters():
            p.requires_grad = False
    
    def extract(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from audio
        
        Args:
            audio_tensor: Audio tensor [1, T]
            
        Returns:
            Content features [1, channels, frames]
        """
        return self.soft_model(audio_tensor)
