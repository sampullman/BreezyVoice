# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import time
from cosyvoice.utils.runtime import (
    build_stage_autocast,
    clear_device_cache,
    configure_torch_runtime,
    get_amp_dtype_name,
    get_torch_device,
    should_log_stage_timings,
)

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        configure_torch_runtime(torch)
        self.device = get_torch_device(torch)
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.flow_amp_dtype = get_amp_dtype_name("flow")
        self.hift_amp_dtype = get_amp_dtype_name("hift")
        self.hift_weight_norm_removed = False

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()
        if hasattr(self.hift, "remove_weight_norm"):
            self.hift.remove_weight_norm()
            self.hift_weight_norm_removed = True

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32)):
        with torch.inference_mode():
            llm_started = time.perf_counter()
            tts_speech_token = self.llm.inference(text=text.to(self.device),
                                                  text_len=text_len.to(self.device),
                                                  prompt_text=prompt_text.to(self.device),
                                                  prompt_text_len=prompt_text_len.to(self.device),
                                                  prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                  prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                  embedding=llm_embedding.to(self.device),
                                                  beam_size=1,
                                                  sampling=25,
                                                  max_token_text_ratio=30,
                                                  min_token_text_ratio=3)
            llm_seconds = time.perf_counter() - llm_started
            flow_started = time.perf_counter()
            with build_stage_autocast(torch, self.device, "flow"):
                tts_mel = self.flow.inference(token=tts_speech_token,
                                              token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                              prompt_token=flow_prompt_speech_token.to(self.device),
                                              prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                              prompt_feat=prompt_speech_feat.to(self.device),
                                              prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                              embedding=flow_embedding.to(self.device))
            flow_seconds = time.perf_counter() - flow_started
            hift_started = time.perf_counter()
            with build_stage_autocast(torch, self.device, "hift"):
                tts_speech = self.hift.inference(mel=tts_mel).cpu()
            hift_seconds = time.perf_counter() - hift_started
        if should_log_stage_timings():
            print(
                "CosyVoice stage timing:",
                {
                    "llm_seconds": round(llm_seconds, 3),
                    "flow_seconds": round(flow_seconds, 3),
                    "hift_seconds": round(hift_seconds, 3),
                    "token_count": int(tts_speech_token.size(1)),
                    "mel_frames": int(tts_mel.shape[-1]),
                    "flow_amp": self.flow_amp_dtype or "off",
                    "hift_amp": self.hift_amp_dtype or "off",
                    "hift_weight_norm_removed": self.hift_weight_norm_removed,
                },
            )
        clear_device_cache(torch, self.device)
        return {'tts_speech': tts_speech}
