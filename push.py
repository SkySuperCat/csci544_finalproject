from transformers import T5ForConditionalGeneration, T5TokenizerFast
from huggingface_hub import HfApi, HfFolder

model_save_path = "/data/local/cat_data/qgmodel2"
tokenizer_save_path = "/data/local/cat_data/qgmodel2"

model = T5ForConditionalGeneration.from_pretrained(model_save_path)
tokenizer = T5TokenizerFast.from_pretrained(tokenizer_save_path)

# Authenticate with your Hugging Face account
hf_folder = HfFolder()
token = hf_folder.get_token()
api = HfApi()

# Change 'your_username' to your Hugging Face username
model_repo_name = "supercat666/qg_en"
tokenizer_repo_name = "supercat666/qg_en"

# Push the model and tokenizer to Hugging Face Hub
model.push_to_hub(model_repo_name, use_auth_token=token)
tokenizer.push_to_hub(tokenizer_repo_name, use_auth_token=token)
