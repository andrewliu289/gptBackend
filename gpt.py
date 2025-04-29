from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import importlib.util
import warnings

class GPTModelHandler:
    def __init__(self, device: str):
        self.model_id = "VishnuT/llama3-merged-phase2.3"
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
                    warnings.warn(f"bitsandbytes not usable -> FP16")
            else:
                warnings.warn("bitsandbytes not installed -> FP16.")

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

    def predict(self, prompt: str, max_new_tokens: int = 100) -> dict:
        try:
            # Tokenize input
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(self.model.device)
            attention_mask = tokens["attention_mask"].to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    repetition_penalty=1.2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode and clean output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_output = decoded_output.strip()

            # Remove prompt from output (optional)
            if decoded_output.startswith(prompt):
                decoded_output = decoded_output[len(prompt):].strip()

            return {"prediction_gpt": decoded_output}
        except Exception as e:
            return {"error": str(e)}
