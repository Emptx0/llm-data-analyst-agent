from transformers import AutoModelForImageTextToText, AutoProcessor
import torch


class LLMEngine:
    def __init__(self, model_id: str, device_map: str = 'auto'):
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                offload_buffers=torch.cuda.is_available(), 
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
    
    def generate(self, messages, max_new_tokens)  -> str:
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

        return llm_output

