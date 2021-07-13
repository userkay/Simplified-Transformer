import torch
from torch import nn, optim
from torch.nn import init, Parameter, functional as F

def contextualise(x, dim=1, fn="cumsum", keepdim=True):
    if fn == "cumsum":
        return getattr(torch, fn)(x, dim=dim)
    return getattr(torch, fn)(x, dim=dim, keepdim=keepdim)

def global_dot_score(query, key):
    # key = context
    # query = sequence
    energy = (query * key).sum(dim=1)
    return energy.unsqueeze(1)*query

def get_global_context(x, dim=1, fn="cumsum", keepdim=True):
    ctx = contextualise(x, dim, fn, keepdim)
    rep = global_dot_score(x, ctx)
    return rep, ctx

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        
    def forward(self, query, keys, values, mask=None):

        scaling = (self.embed_size ** (1/2))
        query = self.feature_map(query)
        keys = self.feature_map(keys)
        
        energy = (global_dot_score(query, keys)/scaling) #if mask is None else (global_dot_score(query, keys)/scaling) * mask
        attention = (torch.softmax(energy, dim=1) * values)
#         attention =  (torch.div(energy, energy.sum(dim=1, keepdim=True)) * values)
        
        return attention
    
    def feature_map(self, x):
        return F.gelu(x) + 1

class CrossAttention(nn.Module):
    def __init__(self, embed_size):
        super(CrossAttention, self).__init__()
        
        self.embed_size = embed_size
        
    def forward(self, query, keys, values, mask=None):
        scaling = (self.embed_size ** (1/2))
        query = self.feature_map(query)
        keys = self.feature_map(keys)
        
        energy = (global_dot_score(query, keys)/scaling) #if mask is None else (global_dot_score(query, keys)/scaling) * mask
        attention = (torch.softmax(energy, dim=1) * values)
#         attention =  (torch.div(energy, energy.sum(dim=1, keepdim=True)) * values)
        return attention
    
    def feature_map(self, x):
        return F.gelu(x) + 1

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, dropout, is_decoder=False):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder
        if is_decoder:
            self.attention = CrossAttention(embed_size)
        else:
            self.attention = SelfAttention(embed_size)
        
    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        return x

class Encoder(nn.Module):
    def __init__(
        self, src_vocab_size, embed_size, num_layers, 
        device, dropout, max_length, 
        pool="cumsum", return_context=True):
        
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
                
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, dropout)
             for _ in range(num_layers)]
        )
        
        self.contextualise = get_global_context
            
        self.dropout = nn.Dropout(dropout)
        self.return_context = return_context
        self.num_layers = num_layers
        self.pool = pool
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        
        out = self.dropout(self.word_embedding(x))
        
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)
                
        out, ctx = self.contextualise(out, dim=1, fn=self.pool)
        
        count = 0
        rep = out.clone()
        for layers in self.layers:
            out = layers(out, ctx, rep, mask)
            count+=1
            if count < self.num_layers:
                rep, ctx = self.contextualise(out, dim=1, fn=self.pool)
            
        if self.return_context:
            if self.pool == "cumsum":
                return out, ctx[:, -1].unsqueeze(1)
            return out, ctx
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, dropout, is_decoder=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key1, value1, key2, value2, src_mask, trg_mask):
        attention = self.attention(x, key1, value1, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, key2, value2, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, 
        dropout, device, max_length, pool="cumsum", return_context=False):
        super(Decoder, self).__init__()
        
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, dropout, device)
             for _ in range(num_layers)]
        )
                
        self.return_context = return_context
        
        self.contextualise = get_global_context
            
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.pool = pool
    
    def forward(self, x, enc_out, enc_ctx, src_mask, trg_mask):
        N, seq_length = x.shape
        x = self.dropout(self.word_embedding(x))
        
        if trg_mask is not None:
            if len(trg_mask.shape) == 2:
                trg_mask = trg_mask.unsqueeze(-1)
            
        x, ctx = self.contextualise(x, dim=1, fn=self.pool)
        
        rep = x.clone()
        count=0
        
        for layer in self.layers:
            x = layer(x, ctx, rep, enc_ctx[:, -1, :].unsqueeze(1), 
                      enc_out.sum(dim=1, keepdim=True), src_mask, trg_mask)
            count+=1
            if count < self.num_layers:
                rep, ctx = self.contextualise(x, dim=1, fn=self.pool)
        if self.return_context:
            if self.pool == "cumsum":
                return x, ctx[:, -1]
            return x, ctx
        return x

    
class Predictor(nn.Module):
    def __init__(self, in_features, out_features, pool='mean', use="seq"):
        super().__init__()
        
        self.pool = pool
        self.use = use
        self.pools = ["mean", "sum", "max", "softmax", "softmax_mean", 
                        "adaptive_avg_pool"]
        assert (use in ['seq', 'ctx', 'both']), \
        f"Value error: \'use\' must be one of {[v for v in use]}"
        if pool not in self.pools:
            assert (pool is None), f"Input Error: pool must be one of {self.pools}"
        if use == "both":
            assert (pool is not None), f"\'pool\' cannot be None for \'use={use}'"
            self.classifier = nn.Linear(2*in_features, out_features)
        else:
            if use == 'seq':
                assert (pool is not None), f"\'pool\' cannot be None for \'use={use}'"
            self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x1, x2=None):
        if self.use == "ctx":
            return self.classifier(x2)
        elif self.use == "both":
            assert x2 is not None
            B, L, D = x1.shape
            if len(x2.shape) == 2:
                ctx = x2.unsqueeze(1)
#             print(ctx.shape)
            ctx = ctx.expand(B, L, D)
            x = torch.cat((x1, ctx), dim=-1)
            out = self.clf_pooler(x, self.pool, dim=1)
        else:
            if self.pool == None:
                return self.classifier(x1)
            else:
                out = self.clf_pooler(x1, self.pool, dim=1)
                        
        out = self.classifier(out)
        return out
    
    def clf_pooler(self, x, pool, dim=1):
        if pool == 'softmax':
            out = torch.nn.Softmax(dim=dim)(x)
            out, _ = torch.max(out, dim=dim)
        elif pool == 'softmax_mean':
            out = torch.nn.Softmax(dim=dim)(x)
            out = torch.mean(out, dim=dim)
        elif pool in ['mean', 'sum']:
            out = getattr(torch, pool)(x, dim=dim)
        elif pool == 'max':
            out = torch.nn.AdaptiveMaxPool2d((1, None))(x).squeeze(1)
        elif pool == 'adaptive_avg_pool':
            out = torch.nn.AdaptiveAvgPool2d((1, None))(x).squeeze(1)
        else:
            out = None
            raise Exception(f"Value Error: Pooling option '{pool}' not defined.")
        return out
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
                 embed_size=128, output_size=2, num_layers=2, 
                 dropout=0.1, device="cpu", max_length=512, pool="cumsum", clf_pool="mean", 
                 same_embeddings=True, return_context=False, use_context="seq"):
        super().__init__()
        
        self.return_context = return_context
        self.use_context = use_context
        if use_context != "seq":
            assert (return_context), "return_context must be set to true with use_context"
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, 
                               device, dropout, max_length, 
                               pool, return_context=True)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, 
                               dropout, device, max_length, 
                               pool, return_context)
        if same_embeddings:
            self.decoder.word_embedding = self.encoder.word_embedding
        self.predictor = Predictor(embed_size, output_size, clf_pool, use_context)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device        
        
    def make_src_mask(self, src):
        pass
#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
    def resize_mask(self, mask):
        return mask.unsqueeze(-1)
        
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg, src_mask=None, trg_mask=None, make_mask=False):
        src_mask = self.make_src_mask(src) if src_mask is None else self.resize_mask(src_mask)
        if trg_mask is None:
            if src_mask is not None:
                trg_mask = src_mask.clone()
            else:
                trg_mask = self.make_src_mask(trg) 
        else: 
            trg_mask = self.resize_mask(trg_mask)
        
        if src_mask is not None: 
            src_mask.requires_grad = False
        if trg_mask is not None: 
            trg_mask.requires_grad = False
            
        enc_src, enc_ctx = self.encoder(src, src_mask)
        if self.decoder.return_context:
            out, ctx = self.decoder(trg, enc_src, enc_ctx, src_mask, trg_mask)
        else:
            out = self.decoder(trg, enc_src, enc_ctx, src_mask, trg_mask)
        if self.use_context != "seq":
            return self.predictor(out, ctx)
        out = self.predictor(out)
        return out
    
def transformer_for_classification(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
                embed_size=128, output_size=2, num_layers=2, 
                dropout=0.0, device="cpu", max_length=100, pool="cumsum", clf_pool=None, 
                same_embeddings=True, return_context=False, use_context='seq'):
    
    return Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, 
                       trg_pad_idx, embed_size, output_size, num_layers, 
                       dropout, device, max_length, pool, clf_pool, 
                       same_embeddings, return_context, use_context).to(device)
