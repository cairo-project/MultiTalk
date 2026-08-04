"""MultiTalk inference: load_models / generate_video interface (single-person mode).

Wraps the MultiTalk pipeline for single-person audio-driven talking video:
    models = load_models(ckpt_dir, wav2vec_dir)
    generate_video(models, source_image, driving_audio, save_path)
"""

import json
import logging
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.multitalk_utils import save_video_ffmpeg
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

logger = logging.getLogger("multitalk")


def _custom_init(device, wav2vec_dir):
    """Initialize wav2vec feature extractor and audio encoder."""
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir)
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    audio_encoder.eval()
    return wav2vec_feature_extractor, audio_encoder


def _get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder):
    """Extract audio embedding from speech array."""
    audio_feature = wav2vec_feature_extractor(
        human_speech, sampling_rate=16000, return_tensors="pt"
    ).input_values.to(audio_encoder.device)
    seq_len = torch.tensor([audio_feature.shape[-1]], device=audio_encoder.device)
    with torch.no_grad():
        embed = audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
    embed = embed.hidden_states
    audio_emb = torch.stack(embed, dim=1).squeeze(0).permute(2, 0, 1)
    return audio_emb


def _audio_prepare_single(audio_path, sample_rate=16000):
    """Load and prepare a single audio track."""
    human_speech, _ = librosa.load(audio_path, sr=sample_rate)
    return human_speech


def load_models(
    ckpt_dir: str,
    wav2vec_dir: str,
    task: str = "multitalk-14B",
    size: str = "multitalk-480",
    device: str = "cuda",
    num_persistent_param_in_dit: int | None = None,
) -> dict:
    """Load all MultiTalk models for single-person inference.

    Args:
        ckpt_dir: Path to Wan2.1-I2V-14B-480P checkpoint dir (with multitalk weights linked in)
        wav2vec_dir: Path to chinese-wav2vec2-base model
        task: Task name (default 'multitalk-14B')
        size: Resolution size ('multitalk-480' or 'multitalk-720')
        device: CUDA device
        num_persistent_param_in_dit: For VRAM management (None = no management)
    """
    cfg = WAN_CONFIGS[task]
    device_id = 0

    logger.info("Loading wav2vec audio encoder...")
    wav2vec_feature_extractor, audio_encoder = _custom_init("cpu", wav2vec_dir)

    logger.info("Creating MultiTalk pipeline...")
    pipeline = wan.MultiTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        quant_dir=None,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        lora_dir=None,
        lora_scales=None,
        quant=None,
    )

    if num_persistent_param_in_dit is not None:
        pipeline.vram_management = True
        pipeline.enable_vram_management(
            num_persistent_param_in_dit=num_persistent_param_in_dit
        )

    return {
        "pipeline": pipeline,
        "wav2vec_feature_extractor": wav2vec_feature_extractor,
        "audio_encoder": audio_encoder,
        "size": size,
        "device": device,
    }


def generate_video(
    models: dict,
    source_image: str,
    driving_audio: str,
    save_path: str,
    prompt: str = "A person is talking.",
    frame_num: int = 81,
    sample_steps: int = 40,
    sample_shift: float = 5.0,
    text_guide_scale: float = 5.0,
    audio_guide_scale: float = 2.5,
    seed: int = 42,
    **kwargs,
) -> str:
    """Generate a single-person talking video from image and audio.

    Args:
        models: Dict from load_models()
        source_image: Path to reference portrait image
        driving_audio: Path to driving audio (WAV)
        save_path: Output video path
        prompt: Text prompt describing the scene
        frame_num: Number of frames to generate
        sample_steps: Diffusion sampling steps
        sample_shift: Sample shift parameter
        text_guide_scale: Text guidance scale
        audio_guide_scale: Audio guidance scale
        seed: Random seed

    Returns:
        Path to generated video
    """
    pipeline = models["pipeline"]
    wav2vec_feature_extractor = models["wav2vec_feature_extractor"]
    audio_encoder = models["audio_encoder"]
    size = models["size"]

    # Prepare audio
    with tempfile.TemporaryDirectory(prefix="multitalk_") as tmpdir:
        human_speech = _audio_prepare_single(driving_audio)
        audio_embedding = _get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)

        emb_path = os.path.join(tmpdir, "1.pt")
        sum_audio = os.path.join(tmpdir, "sum.wav")
        sf.write(sum_audio, human_speech, 16000)
        torch.save(audio_embedding, emb_path)

        # Build input_data for single-person mode
        input_data = {
            "cond_image": source_image,
            "cond_audio": {"person1": emb_path},
            "prompt": prompt,
            "video_audio": sum_audio,
        }

        logger.info("Generating video...")
        video = pipeline.generate(
            input_data,
            size_buckget=size,
            motion_frame=0,
            frame_num=frame_num,
            shift=sample_shift,
            sampling_steps=sample_steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            offload_model=False,
            max_frames_num=frame_num,
            color_correction_strength=0.0,
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video_ffmpeg(video, save_path.replace(".mp4", ""), [sum_audio], high_quality_save=False)

    return save_path
