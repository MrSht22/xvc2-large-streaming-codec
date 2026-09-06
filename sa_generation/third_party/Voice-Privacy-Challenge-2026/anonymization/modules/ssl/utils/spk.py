import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from kaldiio import ReadHelper
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import resampy

class SpeakerEmbeddingExtractor:
    """ECAPA-TDNN based speaker embedding extractor"""
    
    def __init__(self, ecapa_ckpt_path: str, device: str = "cpu"):
        self.device = device
        
        # Initialize ECAPA-TDNN
        self.ecapa = ECAPA_TDNN(input_size=80, lin_neurons=192)
        
        # Load checkpoint
        ckpt = torch.load(ecapa_ckpt_path, map_location=device)
        self.ecapa.load_state_dict(ckpt)
        self.ecapa.to(device).eval()
        
        # Freeze parameters
        for p in self.ecapa.parameters():
            p.requires_grad = False
        
        # Initialize feature extractors
        self.fbank_extractor = Fbank(n_mels=80)
        self.norm_proc = InputNormalization(norm_type="sentence", std_norm=False)
    
    def extract(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract speaker embedding from audio
        
        Args:
            audio_tensor: Audio tensor [1, T]
            
        Returns:
            Speaker embedding vector [D]
        """
        wav = audio_tensor.to(self.device)
        
        # Extract features
        feats = self.fbank_extractor(wav)
        feats = self.norm_proc(feats, torch.ones(feats.size(0), device=self.device))
        
        # Get embedding
        out = self.ecapa(feats)
        emb = out[0] if isinstance(out, (tuple, list)) else out
        
        # Process dimensions
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        if emb.ndim == 2:
            emb = emb[0]
        else:
            emb = emb.reshape(emb.shape[0], -1)[0]
        
        # Normalize
        emb = emb / (torch.norm(emb, p=2) + 1e-8)
        
        return emb.detach().cpu().numpy().astype(np.float32)

class SpeakerPoolManager:
    """Manager for speaker pool operations"""
    
    @staticmethod
    def load_xvectors_from_scp(xvec_file: str) -> Dict[str, np.ndarray]:
        """Load xvectors from Kaldi scp file"""
        xvector_dic = {}
        # Get the absolute path of the SCP file BEFORE changing directories
        abs_xvec_file = os.path.abspath(xvec_file)
        scp_dir = os.path.dirname(abs_xvec_file)
        old_cwd = os.getcwd()
        
        try:
            # Change to the directory containing the SCP file
            os.chdir(scp_dir)
            
            # If the scp file contains 'selec_anon/', we need to find the root where it exists
            # so kaldiio can resolve relative paths inside the SCP
            for _ in range(5):
                if os.path.exists("selec_anon"):
                    break
                os.chdir("..")
            
            # Use the pre-calculated absolute path
            with ReadHelper("scp:" + abs_xvec_file) as reader:
                for key, xvec in reader:
                    xvec = np.asarray(xvec)
                    if xvec.ndim == 3 and xvec.shape[0] == 1:
                        xvec = xvec.squeeze(0)
                    if xvec.ndim == 2 and xvec.shape[0] == 1:
                        xvec = xvec.squeeze(0)
                    xvector_dic[key] = xvec.astype(np.float32)
        finally:
            os.chdir(old_cwd)
        
        print(f"Loaded {len(xvector_dic)} xvectors from pool.")
        return xvector_dic
    
    @staticmethod
    def _norm_gender(g: str) -> Optional[str]:
        """Normalize gender string"""
        g = str(g).strip().lower() if g is not None else None
        if g in {"m", "male"}:
            return "m"
        if g in {"f", "female"}:
            return "f"
        return None
    
    @staticmethod
    def _maybe_match_speaker(key: str, spk2gender: Dict[str, str]) -> Optional[str]:
        """Try exact, else prefix before '-' or '_'."""
        if key in spk2gender:
            return key
        for sep in ("-", "_"):
            if sep in key:
                cand = key.split(sep)[0]
                if cand in spk2gender:
                    return cand
        return None
    
    @staticmethod
    def load_spk2gender(path: str) -> Dict[str, str]:
        """Load speaker to gender mapping"""
        spk2gender = {}
        if not os.path.isfile(path):
            print(f"[WARN] spk2gender not found at: {path}. Gender filtering disabled.")
            return spk2gender
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spk, g = parts[0], SpeakerPoolManager._norm_gender(parts[1])
                    if g in {"m", "f"}:
                        spk2gender[spk] = g
        
        print(f"Loaded {len(spk2gender)} genders from {path}.")
        return spk2gender
    
    def __init__(self, xvector_scp_path: str, gender_map_path: str):
        """Initialize speaker pool manager"""
        # Load raw data
        self.raw_xvector_pool = self.load_xvectors_from_scp(xvector_scp_path)
        self.spk2gender = self.load_spk2gender(gender_map_path)
        
        # Prepare matrices
        self._prepare_pool_matrices()
    
    def _prepare_pool_matrices(self):
        """Prepare normalized matrices for fast similarity computation"""
        keys = []
        vecs = []
        
        for k, v in self.raw_xvector_pool.items():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 3 and v.shape[0] == 1:
                v = v.squeeze(0)
            if v.ndim == 2 and v.shape[0] == 1:
                v = v.squeeze(0)
            if v.ndim != 1:
                v = v.reshape(-1)
            vecs.append(v)
            keys.append(k)
        
        if len(vecs) == 0:
            raise ValueError("Empty xvector pool after loading.")
        
        # Stack vectors
        X = np.stack(vecs).astype(np.float32)  # (N, D)
        
        # Check dimension
        if X.shape[1] != 192:
            print(f"[WARN] xvector dim is {X.shape[1]}, expected 192 for ECAPA; continuing anyway.")
        
        # L2 normalize for cosine similarity
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        # Save full pool
        self.pool_keys_all = np.array(keys)
        self.pool_X_all = X
        self.pool_Xn_all = X_norm
        
        # Precompute gender masks
        gmap = self.spk2gender
        m_mask = np.array([(gmap.get(self._maybe_match_speaker(k, gmap), None) == "m") 
                          for k in keys])
        f_mask = np.array([(gmap.get(self._maybe_match_speaker(k, gmap), None) == "f") 
                          for k in keys])
        
        self.pool_mask_m = m_mask
        self.pool_mask_f = f_mask
    
    def get_pool_view(self, gender: Optional[str], gender_pool: bool = True) -> Tuple:
        """
        Get appropriate pool view based on gender
        
        Args:
            gender: Target gender ('m', 'f', or None)
            gender_pool: Whether to filter by gender
            
        Returns:
            Tuple of (keys, vectors, normalized_vectors)
        """
        if not gender_pool:
            # Use entire pool
            return self.pool_keys_all, self.pool_X_all, self.pool_Xn_all
        
        gender = self._norm_gender(gender)
        
        if gender == "m" and self.pool_mask_m.any():
            return (self.pool_keys_all[self.pool_mask_m], 
                    self.pool_X_all[self.pool_mask_m], 
                    self.pool_Xn_all[self.pool_mask_m])
        
        if gender == "f" and self.pool_mask_f.any():
            return (self.pool_keys_all[self.pool_mask_f], 
                    self.pool_X_all[self.pool_mask_f], 
                    self.pool_Xn_all[self.pool_mask_f])
        
        # Fallback to full pool
        if gender in {"m", "f"}:
            print(f"[WARN] No xvectors after gender='{gender}' filter. Falling back to full pool.")
        
        return self.pool_keys_all, self.pool_X_all, self.pool_Xn_all


class GenderDetector:
    """Gender detection using Wav2Vec2 model"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Load model and processor
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "prithivMLmods/Common-Voice-Geneder-Detection"
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "prithivMLmods/Common-Voice-Geneder-Detection"
        )
        
        # Gender mapping
        self.id2label = {
            "0": "female",
            "1": "male"
        }
        
        self.model.to(device).eval()
        
        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False
    
    def detect(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        Detect gender from audio
        
        Args:
            audio_array: Audio array [samples] or [samples, channels]
            sample_rate: Original sampling rate
            
        Returns:
            Gender label ('m' or 'f')
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_array = resampy.resample(audio_array, sample_rate, 16000)
        
        # Ensure 1D audio
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) 
                  for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
        
        # Handle single value case
        if not isinstance(probs, list):
            probs = [probs]
        
        # Get predictions
        prediction = {
            self.id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
        }
        
        # Return gender
        if prediction.get("male", 0) > prediction.get("female", 0):
            return "m"
        else:
            return "f"
