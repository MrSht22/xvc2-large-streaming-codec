ARCHIVE_NAMES = ("anonymization.zip", "asr.zip", "tts.zip")

EXPECTED_ARCHIVE_SIZES = {
    "anonymization.zip": 850894,
    "asr.zip": 408472040,
    "tts.zip": 748014864,
}

EXPECTED_ARCHIVE_SHA256 = {
    "anonymization.zip": "0e662a713fb3f01b25000b7d1fa65ff5dd58a39ed7922703df26c8b7a9ba7e32",
    "asr.zip": "14dd433addca14a57a67d1ad838c830ea1a5cb4070dc6a5be95b14ab7a86133d",
    "tts.zip": "c9a40069693d98c4d7a9c17646aa670ded668c0f0cc376175fc26d4fb706b8c1",
}

REQUIRED_STTTS_MODELS = (
    "anonymization/gan_style-embed/style-embed_wgan.pt",
    "asr/asr_branchformer_tts-phn_en.zip",
    "tts/Aligner/aligner.pt",
    "tts/Embedding/embedding_function.pt",
    "tts/FastSpeech2_Multi/prosody_cloning.pt",
    "tts/HiFiGAN_combined/best.pt",
)
