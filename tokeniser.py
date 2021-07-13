from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def bert_tokenizer(filepath, max_len=None, bos_eos_tokens=None):
    tokenizer = Tokenizer.from_file(filepath)
    tokenizer.pre_tokenizer = Whitespace()
    
    if bos_eos_tokens:
        if bos_eos_tokens == "bert_default":
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", tokenizer.token_to_id("[CLS]")),
                    ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ],
            )
            
        else:
            bos=bos_eos_tokens[0]
            eos=bos=bos_eos_tokens[1]
            tokenizer.post_processor = TemplateProcessing(
                single=f"{bos} $A {eos}",
                pair=f"{bos} $A {eos} $B:1 {eos}:1",
                special_tokens=[
                    (f"{bos}", tokenizer.token_to_id(f"{bos}")),
                    (f"{eos}", tokenizer.token_to_id(f"{eos}")),
                ],
            )
            
    if max_len:
        bert_padding(tokenizer, max_len)
    return tokenizer

def bert_padding(tok, max_len=512, pad_token="[PAD]"):
    padding_id = tok.token_to_id(pad_token)
    tok.enable_padding(pad_id=padding_id, pad_token=pad_token, length=max_len)
    tok.enable_truncation(max_length=max_len)
