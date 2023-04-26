from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

model_repo_name = "supercat666/qg"
tokenizer_repo_name = "supercat666/qg"

model = T5ForConditionalGeneration.from_pretrained(model_repo_name)
tokenizer = T5TokenizerFast.from_pretrained(tokenizer_repo_name)

device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def zh_run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  input_string = '%s </s>' % (input_string)
  input_ids = tokenizer.encode(input_string, return_tensors="pt")
  res = model.generate(input_ids.to(device), **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  return output

output = zh_run_model('answer:第二大经济体  context: 近年来，中国经济快速发展，已成为世界第二大经济体。')
print(output)
