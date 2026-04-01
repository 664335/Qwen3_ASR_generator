# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from .qwen3_forced_aligner import Qwen3ForcedAligner
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except:
    pass


@dataclass
class ASRTranscription:
    """
    One transcription result.

    Attributes:
        language (str):
            Merged language string for the sample, e.g. "Chinese" or "Chinese,English".
            Empty string if unknown or silent audio.
        text (str):
            Transcribed text.
        time_stamps (Optional[Any]):
            Forced aligner output (ForcedAlignResult).
            Present only when return_time_stamps=True.
    """
    language: str
    text: str
    time_stamps: Optional[Any] = None


@dataclass
class ASRTranscriptionChunk:
    """
    Single audio chunk transcription result for streaming.

    Attributes:
        chunk_index (int):
            Global chunk index across all audios.
        orig_index (int):
            Original audio index in the input batch.
        text (str):
            Transcribed text for this chunk.
        language (str):
            Detected language for this chunk.
        offset_sec (float):
            Start offset of this chunk in the original audio (seconds).
        duration_sec (float):
            Duration of this chunk (seconds).
        time_stamps (Optional[Any]):
            Forced aligner output for this chunk (if enabled).
    """
    chunk_index: int
    orig_index: int
    text: str
    language: str
    offset_sec: float
    duration_sec: float
    time_stamps: Optional[Any] = None


@dataclass
class ASRTranscriptionInfo:
    """
    Final transcription summary info (returned by generator).

    Attributes:
        language (str or List[str]):
            Merged language(s). str for single audio, List[str] for batch.
        text (str or List[str]):
            Merged full text(s). str for single audio, List[str] for batch.
        duration (float):
            Total audio duration in seconds.
        time_stamps (Optional[Any]):
            Merged forced aligner output (if enabled).
    """
    language: Union[str, List[str]]
    text: Union[str, List[str]]
    duration: float
    time_stamps: Optional[Any] = None


@dataclass
class ASRStreamingState:
    """
    Streaming ASR state for one audio stream (single utterance).

    Attributes:
        unfixed_chunk_num (int):
            For the first N chunks, do not use previous ASR result as prefix prompt (reset prefix to "").
        unfixed_token_num (int):
            When chunk_id >= unfixed_chunk_num, rollback the last K tokens from the accumulated text
            before using it as prefix prompt, to reduce boundary jitter.
        chunk_size_sec (float):
            Chunk size in seconds. Audio will be fed to the model in increments of this length.
        chunk_size_samples (int):
            Chunk size in samples at 16kHz (derived from chunk_size_sec).
        chunk_id (int):
            Current chunk index (0-based).
        buffer (np.ndarray):
            Buffered PCM samples that are not yet consumed into a full chunk.
        audio_accum (np.ndarray):
            Accumulated audio from the beginning of the stream up to current time (no padding).
        prompt_raw (str):
            Base prompt generated by chat template (with generation prompt), without appended prefix text.
        context (str):
            Context string.
        force_language (Optional[str]):
            If provided, force output to be text-only by appending "language X<asr_text>" in prompt_raw,
            consistent with non-streaming transcribe().
        language (str):
            Latest parsed language (updated after each chunk decode). Empty if unknown/silent.
        text (str):
            Latest parsed transcription text (updated after each chunk decode).
        _raw_decoded (str):
            Internal accumulated decoded raw text (before parse_asr_output normalization).
            Used for rollback/token trimming and as prefix for prompting.
    """
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str


class Qwen3ASRModel:
    """
    Unified inference wrapper for Qwen3-ASR with two backends:
      - Transformers backend 
      - vLLM backend

    It optionally supports time stamp output via Qwen3-ForcedAligner.

    Notes:
      - Each request uses a context text and exactly one audio.
      - If language is provided, the prompt will force the output to be text-only by appending
        "language {Language}<asr_text>" to the assistant prompt.
    """

    def __init__(
        self,
        backend: str,
        model: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        forced_aligner: Optional[Qwen3ForcedAligner] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
    ):
        self.backend = backend  # "transformers" | "vllm"
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.forced_aligner = forced_aligner
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens

        if backend == "transformers":
            self.device = getattr(model, "device", None)
            if self.device is None:
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")
            self.dtype = getattr(model, "dtype", torch.float32)
        else:
            self.device = None
            self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: Optional[int] = 512,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using Transformers backend.

        Args:
            pretrained_model_name_or_path:
                HuggingFace repo id or local directory.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to AutoModel.from_pretrained(...).

        Returns:
            Qwen3ASRModel
        """

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)

        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="transformers",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @classmethod
    def LLM(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using vLLM backend.

        Import is isolated to keep vLLM optional.

        Args:
            model:
                Model path/repo for vLLM.
            forced_aligner:
                Optional forced aligner model path/repo id.
            forced_aligner_kwargs:
                Optional kwargs forwarded to Qwen3ForcedAligner.from_pretrained(...).
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to vllm.LLM(...).

        Returns:
            Qwen3ASRModel

        Raises:
            ImportError: If vLLM is not installed.
        """
        try:
            from vllm import LLM as vLLM
            from vllm import SamplingParams
        except Exception as e:
            raise ImportError(
                "vLLM is not available. Install with: pip install qwen-asr[vllm]"
            ) from e

        llm = vLLM(model=model, **kwargs)

        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))

        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=None,
        )

    def get_supported_languages(self) -> List[str]:
        """
        Returns the supported language list.

        Returns:
            List[str]: Canonical language names.
        """
        return list(SUPPORTED_LANGUAGES)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio with optional context and optional forced alignment timestamps.

        This method internally uses transcribe_streaming() for memory-efficient processing,
        but returns results as a list for backward compatibility.

        Args:
            audio:
                Audio input(s). Supported:
                  - str: local path / URL / base64 data url
                  - (np.ndarray, sr)
                  - list of above
            context:
                Context string(s). If scalar, it will be broadcast to batch size.
            language:
                Optional language(s). If provided, it must be in supported languages.
                If scalar, it will be broadcast to batch size.
                If provided, the prompt will force output to be transcription text only.
            return_time_stamps:
                If True, timestamps are produced via forced aligner and merged across chunks.
                This requires forced_aligner initialized.

        Returns:
            List[ASRTranscription]: One result per input audio.

        Raises:
            ValueError:
                - If return_time_stamps=True but forced_aligner is not provided.
                - If language is unsupported.
                - If batch sizes mismatch for context/language.
        """
        # Use streaming generator internally for memory efficiency
        gen = self.transcribe_streaming(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

        # Consume all chunks to collect results
        per_audio_results: Dict[int, Dict[str, List[Any]]] = {}
        
        for chunk in gen:
            if chunk.orig_index not in per_audio_results:
                per_audio_results[chunk.orig_index] = {
                    "langs": [],
                    "texts": [],
                    "aligns": []
                }
            
            per_audio_results[chunk.orig_index]["langs"].append(chunk.language)
            per_audio_results[chunk.orig_index]["texts"].append(chunk.text)
            if chunk.time_stamps is not None:
                per_audio_results[chunk.orig_index]["aligns"].append(chunk.time_stamps)

        # Build final results
        results: List[ASRTranscription] = []
        for i in sorted(per_audio_results.keys()):
            data = per_audio_results[i]
            merged_text = "".join([t for t in data["texts"] if t is not None])
            merged_language = merge_languages(data["langs"])
            merged_align = self._merge_align_results(data["aligns"]) if return_time_stamps else None
            results.append(ASRTranscription(
                language=merged_language,
                text=merged_text,
                time_stamps=merged_align
            ))

        return results

    @torch.no_grad()
    def transcribe_streaming(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> Generator[ASRTranscriptionChunk, None, ASRTranscriptionInfo]:
        """
        Transcribe audio with streaming output (generator).

        This method yields ASRTranscriptionChunk for each audio chunk as it's processed,
        allowing real-time processing and reduced memory usage.

        Args:
            audio:
                Audio input(s). Supported:
                  - str: local path / URL / base64 data url
                  - (np.ndarray, sr)
                  - list of above
            context:
                Context string(s). If scalar, it will be broadcast to batch size.
            language:
                Optional language(s). If provided, it must be in supported languages.
                If scalar, it will be broadcast to batch size.
                If provided, the prompt will force output to be transcription text only.
            return_time_stamps:
                If True, timestamps are produced via forced aligner.
                This requires forced_aligner initialized.

        Yields:
            ASRTranscriptionChunk: Chunk-level transcription result.

        Returns:
            ASRTranscriptionInfo: Final summary info (accessible via StopIteration.value).

        Raises:
            ValueError:
                - If return_time_stamps=True but forced_aligner is not provided.
                - If language is unsupported.
                - If batch sizes mismatch for context/language.
        """
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner` to be provided at initialization.")

        wavs = normalize_audios(audio)
        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # Accumulators for final merge
        per_audio_langs: List[List[str]] = [[] for _ in range(n)]
        per_audio_texts: List[List[str]] = [[] for _ in range(n)]
        per_audio_aligns: List[List[Any]] = [[] for _ in range(n)]

        # Process chunks one by one and yield
        for chunk_idx, chunk in enumerate(chunks):
            chunk_ctx = [ctxs[chunk.orig_index]]
            chunk_lang = [langs_norm[chunk.orig_index]]
            chunk_wavs = [chunk.wav]

            # Single chunk ASR inference
            raw_output = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)[0]
            lang, txt = parse_asr_output(raw_output, user_language=chunk_lang[0])

            # Single chunk alignment (optional)
            align_result = None
            if return_time_stamps and txt.strip():
                align_result = self.forced_aligner.align(
                    audio=(chunk.wav, chunk.sr),
                    text=txt,
                    language=lang
                )[0]
                align_result = self._offset_align_result(align_result, chunk.offset_sec)

            # Accumulate results
            per_audio_langs[chunk.orig_index].append(lang)
            per_audio_texts[chunk.orig_index].append(txt)
            if align_result is not None:
                per_audio_aligns[chunk.orig_index].append(align_result)

            # Yield current chunk result
            yield ASRTranscriptionChunk(
                chunk_index=chunk_idx,
                orig_index=chunk.orig_index,
                text=txt,
                language=lang,
                offset_sec=chunk.offset_sec,
                duration_sec=len(chunk.wav) / chunk.sr,
                time_stamps=align_result
            )

            # Clean up GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Return final summary info
        final_results = []
        for i in range(n):
            merged_text = "".join([t for t in per_audio_texts[i] if t is not None])
            merged_language = merge_languages(per_audio_langs[i])
            merged_align = self._merge_align_results(per_audio_aligns[i]) if return_time_stamps else None
            final_results.append(ASRTranscription(
                language=merged_language,
                text=merged_text,
                time_stamps=merged_align
            ))

        return ASRTranscriptionInfo(
            language=final_results[0].language if n == 1 else [r.language for r in final_results],
            text=final_results[0].text if n == 1 else [r.text for r in final_results],
            duration=sum(len(w) / SAMPLE_RATE for w in wavs),
            time_stamps=final_results[0].time_stamps if n == 1 else [r.time_stamps for r in final_results]
        )

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        Build the string prompt for one request.

        If force_language is provided, "language X<asr_text>" is appended after the generation prompt
        to request text-only output.
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run backend inference for chunk-level items.

        Args:
            contexts: List of context strings.
            wavs: List of mono waveforms (np.ndarray).
            languages: List of forced languages or None.

        Returns:
            List[str]: Raw decoded strings (one per chunk).
        """
        if self.backend == "transformers":
            return self._infer_asr_transformers(contexts, wavs, languages)
        if self.backend == "vllm":
            return self._infer_asr_vllm(contexts, wavs, languages)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        outs: List[str] = []

        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))

        return outs

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        inputs: List[Dict[str, Any]] = []
        for c, w, fl in zip(contexts, wavs, languages):
            prompt = self._build_text_prompt(context=c, force_language=fl)
            inputs.append({"prompt": prompt, "multi_modal_data": {"audio": [w]}})

        outs: List[str] = []
        for batch in chunk_list(inputs, self.max_inference_batch_size):
            outputs = self.model.generate(batch, sampling_params=self.sampling_params, use_tqdm=False)
            for o in outputs:
                outs.append(o.outputs[0].text)
        return outs

    def _offset_align_result(self, result: Any, offset_sec: float) -> Any:
        """
        Apply time offset to a ForcedAlignResult-like object.

        This function assumes:
          - result has attribute `.items` which is a list of items with start_time/end_time in seconds.
          - dataclasses are frozen in upstream implementation, so we reconstruct by type.

        Args:
            result: ForcedAlignResult
            offset_sec: Offset in seconds

        Returns:
            ForcedAlignResult: New object with shifted timestamps.
        """
        if result is None:
            return None
        items = []
        for it in result.items:
            items.append(type(it)(text=it.text, 
                                  start_time=round(it.start_time + offset_sec, 3), 
                                  end_time=round(it.end_time + offset_sec, 3)))
        return type(result)(items=items)

    def _merge_align_results(self, results: List[Any]) -> Optional[Any]:
        """
        Merge multiple ForcedAlignResult objects into a single one by concatenating items.

        Args:
            results: List of ForcedAlignResult

        Returns:
            ForcedAlignResult or None
        """
        if not results:
            return None
        all_items = []
        for r in results:
            if r is None:
                continue
            all_items.extend(list(r.items))
        if not all_items:
            return None
        return type(results[0])(items=all_items)

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> ASRStreamingState:
        """
        Initialize streaming ASR state for a single stream.

        Notes:
            - Streaming ASR is supported ONLY for vLLM backend.
            - Streaming ASR does NOT support timestamps (forced aligner is not used).
            - Batch inference is NOT supported.

        Args:
            context:
                Context string.
            language:
                Optional forced language. If provided, it must be in supported languages.
                Same behavior as transcribe(): forces text-only output via prompt suffix.
            unfixed_chunk_num:
                For the first N chunks, do not use previous output as prefix prompt (reset prefix to "").
            unfixed_token_num:
                Roll back the last K tokens from accumulated output when using it as prefix prompt
                after unfixed_chunk_num.
            chunk_size_sec:
                Chunk size in seconds (audio is 16k PCM). The function will internally convert it
                to sample count at 16kHz.

        Returns:
            ASRStreamingState: Mutable state object to be passed to streaming_transcribe() and
            finish_streaming_transcribe().

        Raises:
            ValueError:
                - If backend is not "vllm".
                - If chunk_size_sec <= 0.
                - If forced language is invalid (same validation rules as transcribe()).
        """
        if self.backend != "vllm":
            raise ValueError("Streaming ASR is supported only for vLLM backend (backend='vllm').")
        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

        force_language = None
        if language is not None and str(language).strip() != "":
            ln = normalize_language_name(str(language))
            validate_language(ln)
            force_language = ln

        chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
        )

    def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
        """
        Streaming ASR decode step.

        This function accepts an arbitrary-length 16k PCM float numpy array (mono).
        It buffers incoming samples, and whenever enough samples are accumulated to form one
        full chunk (chunk_size_sec), it runs one incremental decode step and updates:

          - state.language
          - state.text

        The caller only needs to keep passing audio to this function and read state.language/state.text.

        Implementation details:
            - Each time a new chunk is ready, we append it to audio_accum and re-feed *all* audio seen
              so far to the model (no padding).
            - We update the prompt as: state.prompt_raw + prefix_text
            - Prefix rollback strategy:
                * If chunk_id < unfixed_chunk_num: prefix_text = ""
                * Else: rollback last unfixed_token_num tokens from previously accumulated decoded text.

        Notes:
            - vLLM backend only.
            - No timestamps.
            - Single stream only (no batching).

        Args:
            pcm16k:
                16kHz mono PCM waveform (np.ndarray). Length can be any non-negative integer.
                dtype can be float32/float64/int16; it will be converted to float32.
            state:
                Streaming state returned by init_streaming_state().

        Returns:
            ASRStreamingState: The same state object (mutated) for convenience.

        Raises:
            ValueError:
                If backend is not "vllm" or state is invalid.
        """
        if self.backend != "vllm":
            raise ValueError("streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        # Ensure 1D mono
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Convert to float32 PCM in [-1, 1] if int16 provided
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32, copy=False)

        # Append to buffer
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Consume full chunks
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            # Accumulate audio (re-feed from start, no padding)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            # Build prefix with rollback strategy
            prefix = ""
            if state.chunk_id < state.unfixed_chunk_num:
                prefix = ""
            else:
                cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
                k = int(state.unfixed_token_num)
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in prefix:
                        break
                    else:
                        if end_idx == 0:
                            prefix = ""
                            break
                        k += 1

            prompt = state.prompt_raw + prefix

            # vLLM input: single item
            inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

            outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
            gen_text = outputs[0].outputs[0].text

            # Accumulate raw decoded (then parse to lang/text)
            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt

            state.chunk_id += 1

        return state

    def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
        """
        Finish streaming ASR.

        This function flushes the remaining buffered audio in state.buffer (tail audio).
        It sends the remaining samples to the model even if shorter than chunk_size_sec,
        without padding. Then it updates state.language/state.text one last time.

        Notes:
            - vLLM backend only.
            - No timestamps.
            - Single stream only.

        Args:
            state:
                Streaming state.

        Returns:
            ASRStreamingState: Updated state (mutated).

        Raises:
            ValueError:
                If backend is not "vllm" or state is invalid.
        """
        if self.backend != "vllm":
            raise ValueError("finish_streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None.")

        # If no remaining buffer, still return state as-is.
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        # Append tail to accumulated audio
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        # Prefix rollback strategy (same as per-chunk)
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

        prompt = state.prompt_raw + prefix
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt

        state.chunk_id += 1
        return state