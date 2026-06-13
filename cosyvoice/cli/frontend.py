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
from functools import partial
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
from cosyvoice.utils.runtime import (
    build_ort_session_options,
    configure_torch_runtime,
    describe_ort_runtime,
    describe_torch_runtime,
    get_torch_device,
    select_ort_providers,
)
try:
    import ttsfrd
    TTSFRD_IMPORT_OK = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    ttsfrd = None
    TTSFRD_IMPORT_OK = False
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph


def _build_text_normalizers():
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer

    return ZhNormalizer(remove_erhua=False, full_to_half=False), EnNormalizer()


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 model_dir: str,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 instruct: bool = False,
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        configure_torch_runtime(torch)
        self.device = get_torch_device(torch)
        option = build_ort_session_options(onnxruntime)
        self.campplus_providers = select_ort_providers(
            onnxruntime,
            preferred_device=torch.device("cpu"),
        )
        self.speech_tokenizer_providers = select_ort_providers(
            onnxruntime,
            preferred_device=self.device,
        )
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model,
            sess_options=option,
            providers=self.campplus_providers,
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=self.speech_tokenizer_providers,
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        use_ttsfrd_env = os.environ.get("BREEZYVOICE_USE_TTSFRD", "auto").strip().lower()
        resource_dir = os.path.join(model_dir, "CosyVoice-ttsfrd", "resource")
        resource_ready = os.path.isdir(resource_dir)
        self.use_ttsfrd = TTSFRD_IMPORT_OK and resource_ready
        if use_ttsfrd_env in ("0", "false", "no"):
            self.use_ttsfrd = False
        elif use_ttsfrd_env in ("1", "true", "yes"):
            if not TTSFRD_IMPORT_OK:
                raise RuntimeError("BREEZYVOICE_USE_TTSFRD requested but ttsfrd is not importable")
            if not resource_ready:
                raise RuntimeError(
                    f"BREEZYVOICE_USE_TTSFRD requested but resource directory is missing: {resource_dir}"
                )
        print(
            "BreezyVoice frontend runtime:",
            {
                "torch": describe_torch_runtime(torch, self.device),
                "campplus_ort": describe_ort_runtime(onnxruntime, self.campplus_providers),
                "speech_tokenizer_ort": describe_ort_runtime(onnxruntime, self.speech_tokenizer_providers),
                "use_ttsfrd": self.use_ttsfrd,
            },
        )
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize(resource_dir) is True, 'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            self.zh_tn_model, self.en_tn_model = _build_text_normalizers()

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None, {self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,]+$', '。', text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        else:
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        if split is False:
            return text
        return texts

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input
