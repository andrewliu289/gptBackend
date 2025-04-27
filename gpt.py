from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GPTModelHandler:
    def __init__(self, device: str, model_id: str = "VishnuT/llama3_qlora_merged_phase_2"):
        self.device = device
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32
        print("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        ).to(device)

        self.model.eval()
        print(f"Model ready on {device}")

    def predict(self, prompt: str) -> dict:
        try:
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = tokens["input_ids"].to(self.model.device)
            attention_mask = tokens["attention_mask"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                early_stopping=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"prediction_gpt": generated_text}
        except Exception as e:
            return {"error": str(e)}
