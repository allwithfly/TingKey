from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ShortcutSettings(BaseModel):
    hotkey: str = "LeftCtrl+LeftWin"
    short_press_action: Literal["speech_assistant", "voice_input", "disabled"] = (
        "speech_assistant"
    )
    long_press_action: Literal["speech_assistant", "voice_input", "disabled"] = (
        "voice_input"
    )
    smart_mouse_mode: bool = False


class MicrophoneSettings(BaseModel):
    device: str = "auto"
    mute_during_recording: bool = False
    recording_dir: str = "recordings"


class SystemSettings(BaseModel):
    auto_start: bool = False
    silent_start: bool = False
    transcription_language: str = "zh-CN"
    enable_logs: bool = False


class LabSettings(BaseModel):
    direct_input_no_clipboard: bool = True
    text_normalization: bool = False


class SkillSettings(BaseModel):
    auto_run: bool = False
    personalize_enabled: bool = False
    personalize_prompt: str = ""
    load_user_dictionary: bool = False
    auto_structure: bool = False
    enhancement_enabled: bool = True
    enhancement_preset: Literal["raw", "voice_input", "code_command"] = "voice_input"
    remove_fillers: bool = True
    auto_punctuation: bool = True
    dedupe_repeats: bool = True
    normalize_spacing: bool = True
    user_dictionary_text: str = ""


class ModelSettings(BaseModel):
    asr_model_size: Literal["0.6b", "1.7b"] = "0.6b"
    asr_model_dir: str = "model"
    asr_backend: Literal["transformers", "vllm"] = "transformers"
    asr_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    asr_max_new_tokens: int = 256
    asr_max_inference_batch_size: int = 8
    asr_attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = (
        "auto"
    )
    asr_enable_tf32: bool = True
    asr_cudnn_benchmark: bool = True
    enable_final_asr: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None


class AppConfig(BaseModel):
    shortcuts: ShortcutSettings = Field(default_factory=ShortcutSettings)
    microphone: MicrophoneSettings = Field(default_factory=MicrophoneSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    lab: LabSettings = Field(default_factory=LabSettings)
    skills: SkillSettings = Field(default_factory=SkillSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
