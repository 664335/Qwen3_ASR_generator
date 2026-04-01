# Qwen3_ASR_generator
modified Qwen3_ASR into a faster-whisper liked generator due to reduce the usage of GPU memory.  
added transcribe_streaming to return timestamps   
```python
import torch
from qwen_asr import Qwen3ASRModel

# 初始化模型（必须指定 forced_aligner）
model = Qwen3ASRModel.from_pretrained(
    pretrained_model_name_or_path = model_path,
    dtype=torch.bfloat16,
    device_map="cuda:0",    
    attn_implementation="flash_attention_2",                 
    max_inference_batch_size=1, # 减小批次大小，降低显存占用
    max_new_tokens=512, # Maximum number of tokens to generate. Set a larger value for long audio input.
    forced_aligner=os.path.join(os.path.dirname(model_path), "Qwen3-ForcedAligner-0.6B"),
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",  
        attn_implementation="flash_attention_2",                          
    )
)

for chunk in model.transcribe_streaming(
    audio = audio_path,
    language = "Chinese",
    return_time_stamps=True
):
    print(f"[块 {chunk.chunk_index}] {chunk.text}")
    
    # 输出该块的时间戳
    if chunk.time_stamps:
        for item in chunk.time_stamps.items:
            print(f"  {item.start_time:.2f}s - {item.end_time:.2f}s: {item.text}")
```
