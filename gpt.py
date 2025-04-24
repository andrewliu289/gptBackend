from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import importlib.util
import warnings
from peft import PeftModel

class GPTModelHandler:
    def __init__(self, device: str):
        self.adapter_path = "VishnuT/llama3-gsm8k-qlora-adapter"
        self.base_model_id = "meta-llama/Llama-3.2-3B"
        self.device = device  # "cuda" or "cpu"

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_kwargs = {}
        can_quantise = False

        if device == "cuda":
            if importlib.util.find_spec("bitsandbytes") is not None:
                try:
                    from bitsandbytes import cuda_setup  # noqa: F401
                    from bitsandbytes import BitsAndBytesConfig

                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    can_quantise = True
                    print("Using bitsandbytes 4-bit NF4 loading …")
                except Exception as e:
                    warnings.warn(
                        f"bitsandbytes not usable ({e}); falling back to FP16."
                    )
            else:
                warnings.warn("bitsandbytes not installed; falling back to FP16.")

        dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading base model")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=None if can_quantise else dtype,
            device_map="auto" if can_quantise else None,
            trust_remote_code=True,
            **quant_kwargs,
        )

        print("Loading QLoRA adapter")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path).to(device)
        self.model.eval()
        print(f"Model ready on {device}")

        # Pipeline config
        pipeline_kwargs = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        if not can_quantise:
            pipeline_kwargs["device"] = 0 if device == "cuda" else -1

        self.generator = pipeline(**pipeline_kwargs)

    def predict(self, prompt: str) -> dict:
        try:
            outputs = self.generator(prompt)
            return {"prediction_gpt": outputs[0]["generated_text"]}
        except Exception as e:
            return {"error": str(e)}
