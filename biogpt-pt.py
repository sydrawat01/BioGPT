import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
m = TransformerLanguageModel.from_pretrained(
        "checkpoints/Pre-trained-BioGPT", 
        "checkpoint.pt", 
        "data",
        tokenizer='moses', 
        bpe='fastbpe', 
        bpe_codes="data/bpecodes",
        min_len=100,
        max_len_b=1024)
prompt = "Empagliflozin is"
src_tokens = m.encode(prompt)
generate = m.generate([src_tokens], beam=5)[0]
output = m.decode(generate[0]["tokens"])
print("Input: ", prompt)
print("Output: ", output)