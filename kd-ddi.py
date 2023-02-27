import torch
from src.transformer_lm_prompt import TransformerLanguageModelPrompt
m = TransformerLanguageModelPrompt.from_pretrained(
        "checkpoints/RE-DDI-BioGPT",
        "checkpoint_avg.pt",
        "data/DDI/relis-bin",
        tokenizer='moses',
        bpe='fastbpe',
        bpe_codes="data/bpecodes",
        max_len_b=1024,
        beam=1)
# m.cuda()
# comment m.cuda() if you are not using the PyTorch with GPU support
src_text="Empagliflozin and Progression of Kidney Disease in Type 2 Diabetes" # input text, e.g., a PubMed abstract
src_tokens = m.encode(src_text)
generate = m.generate([src_tokens], beam=1)[0]
output = m.decode(generate[0]["tokens"])
print(output)