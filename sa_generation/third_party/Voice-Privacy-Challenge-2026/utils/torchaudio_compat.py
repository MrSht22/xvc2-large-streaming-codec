"""
Compatibility patch for torchaudio 2.8+ with speechbrain 1.0.3
Some torchaudio versions removed list_audio_backends() which speechbrain still uses
"""
import torchaudio

# Patch torchaudio.list_audio_backends if it doesn't exist
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        """
        Compatibility function for torchaudio versions without list_audio_backends()
        Returns a list of available backends (newer versions handle backends automatically)
        """
        return ['soundfile', 'sox', 'ffmpeg']
    
    torchaudio.list_audio_backends = list_audio_backends
