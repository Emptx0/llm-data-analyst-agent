from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

import time


class LLMEngine:
    def __init__(self, model_id: str, device_map: str = 'auto'):
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                offload_buffers=torch.cuda.is_available(), 
                dtype=torch.float16
                )
    
    def generate(self, messages: list[dict], max_new_tokens: int)  -> str:
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
           output = self.model.generate(
           **inputs,
           max_new_tokens=max_new_tokens,
           )
    
        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_len:]

        llm_output = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # Some metrics
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        latency = time.time() - start_time
        tokens_sec = len(generated_tokens) / max(latency, 1e-6)
        
        return {
            "output": llm_output,
            "latency": latency,
            "tokens_per_sec": tokens_sec,
            "max_memory_allocated": max_memory_allocated
        }

