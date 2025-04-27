from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import importlib.util
import warnings

class GPTModelHandler:
    def __init__(self, device: str):
        self.model_id = "VishnuT/llama3_qlora_merged_phase_2"
        self.device = device

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # BitsAndBytes
        quant_kwargs = {}
        can_quantise = False
        if device == "cuda":
            if importlib.util.find_spec("bitsandbytes") is not None:
                try:
                    from bitsandbytes import BitsAndBytesConfig
                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    can_quantise = True
                    print("Using bitsandbytes 4-bit NF4 loading")
                except Exception as e:
                    warnings.warn(f"bitsandbytes not usable ({e}); falling back to FP16.")
            else:
                warnings.warn("bitsandbytes not installed; falling back to FP16.")

        dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=None if can_quantise else dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            **quant_kwargs,
        )
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
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"prediction_gpt": generated_text}
        except Exception as e:
            return {"error": str(e)}
