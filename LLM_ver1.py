import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class LMconfig:
    max_context_length: int = 512
    embedding_size: int = 768
    num_blocks: int = 12
    num_attention_heads: int = 12
    head_size: int = embedding_size // num_attention_heads
    dropout_rate: float = 0.2

class SingleAttentionHead(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.query = nn.Linear(config.embedding_size, config.head_size)
        self.key = nn.Linear(config.embedding_size, config.head_size)
        self.value = nn.Linear(config.embedding_size, config.head_size)
        self.head_size = config.head_size

        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.max_context_length, config.max_context_length)
            )
        )
        self.dropout = nn.dropout(config.dropout_rate)

    def forward(self, x):
        _, seq_length, _ = x.size() # (batch, seq, embbedding_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weight = q @ k.transpose(-2, -1) / torch.sqrt(self.head_size)
        weight.masked_fill(
            self.attention_mask[:seq_length, :seq_length] == 0,
            float('-inf')
        )

        weight = F.softmax(weight)
        weight = self.dropout(weight) # why???
        output = weight @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.heads = nn.ModuleList(
            [
                SingleAttentionHead(config)
                for _ in range(config.num_attention_heads)
            ]
        )

        self.proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.net = nn.sequential(
            nn.Linear(config.embedding_size, 4*config.embedding_size),
            nn.GELU(),
            nn.Linear(4*config.embedding_size, config.embedding_size),
            nn.dropout(config.dropout_rate)
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
class block(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm = nn.LayerNorm(config.embedding_size)

    def forward(self, x):
        output = self.norm(x + self.attn(x))
        output = self.norm(output + self.attn(output))
        return output
    
class GPT(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.blocks = nn.Sequential(
            *[block(config) for _ in range(config.num_blocks)]
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.max_context_length, config.embedding_size)

        self.norm_final = nn.LayerNorm(config.embedding_size)
        self.ffn_final = nn.Linear(config.embedding_size, config.vocab_size, bias = False)

        self.apply(self._weight_init)
        
    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.02)

    def forward(self, x, targets = None):
        batch, seq_length = x.size()
        token_embedding = self.token_embedding(x)
        pos_embedding = self.position_embedding(
            torch.arange(
                seq_length, device=x.device
            )
        )

        embedding = token_embedding + pos_embedding
        output = self.blocks(embedding)
        output = self.norm_final(output)
        logits = self.ffn_final(output)

        if targets is None:
            loss = None
        else:
            logits.view(batch*seq_length, -1)
            targets.view(batch, seq_length)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self):
        pass # todo