from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch

class GPTModelHandler:
    def __init__(self, device):
        self.adapter_path = "VishnuT/llama3-gsm8k-qlora-adapter"
        self.base_model_id = "meta-llama/Llama-3.2-3B"
        self.device = device

        print("Loading tokenizer for model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading adapter model for GPT")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(base_model, self.adapter_path).to(self.device)
        self.model.eval()
        print("GPT Model loaded on", self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            device=self.device if self.device == "cuda" else "cpu"
        )

    def predict(self, prompt):
        try:
            outputs = self.generator(prompt)
            return {"prediction_gpt": outputs[0]["generated_text"]}
        except Exception as e:
            return {"error": str(e)}
