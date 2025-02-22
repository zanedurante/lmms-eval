import os
import torch
from typing import List, Tuple, Optional, Union
from transformers import AutoConfig
from prismatic import load
from prismatic.overwatch import initialize_overwatch
from dataclasses import dataclass
from pathlib import Path
import draccus
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from PIL import Image
import torchvision.transforms.functional as TF
import decord
import numpy as np
from tqdm import tqdm

decord.bridge.set_bridge('torch')

@dataclass
class GenerateConfig:
    model_path: Union[str, Path] = "/home/v-zadurante/code/eval_ckpts/4-frames-4-clusters/checkpoints/latest-checkpoint.pt"
    hf_token: Union[str, Path] = Path(".hf_token")
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

@register_model("prismatic_vlm")
class PrismaticVLM(lmms):
    def __init__(
        self,
        pretrained: str = "/home/v-zadurante/code/eval_ckpts/4-frames-4-clusters/checkpoints/latest-checkpoint.pt",
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[int] = 1,
        num_frames: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.num_frames = num_frames
        self.cfg = GenerateConfig()
        self.cfg.model_path = pretrained
        self._model = self._set_up_prismatic_vlm()
        self._model.to(self._device, dtype=torch.bfloat16)
        self._model.eval()
        self.prompt_template = "In: {}\nOut: "
        self._model._supports_cache_class = False

    def _set_up_prismatic_vlm(self):
        overwatch = initialize_overwatch(__name__)
        overwatch.info(f"Initializing Prismatic VLM from {self.cfg.model_path}")
        model = load(self.cfg.model_path)
        model.to(self._device, dtype=torch.bfloat16)
        model.eval()
        return model

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def sample_frames(self, video_path: str):
        vr = decord.VideoReader(video_path)
        frame_indices = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        return [TF.to_pil_image(frames[i]) for i in range(self.num_frames)]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood evaluation is not supported for Prismatic VLM")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([req.args for req in requests]):
            video_path = doc_to_visual(self.task_dict[task][split][doc_id])[0] # Hardcode for now! TODO: Why is this [0] instead of not having [0]?
            assert os.path.exists(video_path), f"Video file {video_path} not found!"
            images = self.sample_frames(video_path)
            prompt_text = self.prompt_template.format(contexts)
            generated_text = self._model.generate(
                images,
                prompt_text,
                do_sample=gen_kwargs.get("do_sample", False),
                temperature=gen_kwargs.get("temperature", 1.0),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
                min_length=gen_kwargs.get("min_length", 1),
            )
            responses.append(generated_text)
        return responses

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Prismatic VLM")