from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained("gpt2", enable_benchmark=True)
print(model.model)
model.compile(num_cores=16)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model.generate(prompts=["Hi there!!"], tokenizer=tokenizer, write_io=True)
