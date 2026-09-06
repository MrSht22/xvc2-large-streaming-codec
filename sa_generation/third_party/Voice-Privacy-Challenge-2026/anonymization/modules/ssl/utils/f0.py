import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

class F0Extractor:
    """F0 extraction using YAAPT algorithm"""
    
    def __init__(self, frame_length: float = 20.0, frame_space: float = 10.0,
                 nccf_thresh1: float = 0.25, tda_frame_length: float = 25.0):
        self.frame_length = frame_length
        self.frame_space = frame_space
        self.nccf_thresh1 = nccf_thresh1
        self.tda_frame_length = tda_frame_length
    
    def extract(self, audio_np: np.ndarray, rate: int = 16000, interp: bool = False) -> np.ndarray:
        """
        Extract F0 from audio
        
        Args:
            audio_np: Audio array [batch, samples] or [samples]
            rate: Sampling rate
            interp: Whether to use interpolated F0
            
        Returns:
            F0 values [batch, 1, frames]
        """
        f0s = []
        
        # Ensure audio_np is 2D [batch, samples]
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        
        for y in audio_np.astype(np.float64):
            to_pad = int(self.frame_length / 1000 * rate) // 2
            y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0.0)
            signal = basic.SignalObj(y_pad, rate)
            pitch = pYAAPT.yaapt(
                signal,
                frame_length=self.frame_length,
                frame_space=self.frame_space,
                nccf_thresh1=self.nccf_thresh1,
                tda_frame_length=self.tda_frame_length,
            )
            vals = pitch.samp_interp if interp else pitch.samp_values
            f0s.append(vals[None, None, :])
        
        return np.vstack(f0s)

