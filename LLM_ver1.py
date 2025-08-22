import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import math
@dataclass
class LMconfig:
    max_context_length: int = 512
    embedding_size: int = 768
    num_blocks: int = 12
    num_attention_heads: int = 12
    head_size: int = embedding_size // num_attention_heads
    dropout_rate: float = 0.2
    vocab_size: int = 50257

class SingleAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        _, seq_length, _ = x.size() # (batch, seq, embbedding_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        weight.masked_fill(
            self.attention_mask[:seq_length, :seq_length] == 0,
            float('-inf')
        )

        weight = F.softmax(weight, dim = -1)
        weight = self.dropout(weight) # why???
        output = weight @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, 4*config.embedding_size),
            nn.GELU(),
            nn.Linear(4*config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm = nn.LayerNorm(config.embedding_size)

    def forward(self, x):
        output = self.norm(x + self.attn(x))
        output = self.norm(output + self.attn(output))
        return output
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.Sequential(
            *[block(config) for _ in range(config.num_blocks)]
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.max_context_length, config.embedding_size)

        self.norm_final = nn.LayerNorm(config.embedding_size)
        self.ffn_final = nn.Linear(config.embedding_size, config.vocab_size, bias = False)

        self.token_embedding.weight = self.ffn_final.weight # tie weight
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
            print(f'before: logits: {logits.shape}, targets: {targets.shape}')
            logits = logits.view(batch * seq_length, -1)
            targets = targets.view(batch * seq_length)
            print(f'after: logits: {logits.shape}, targets: {targets.shape}')
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self):
        pass # todo

class myDataset(Dataset):
    def __init__(self, texts, config):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos = self.enc.encode(
            "<|endoftext|>", 
            allowed_special={"<|endoftext|>"}
        )[0]
        self.encoded_data = []
        block_size = config.max_context_length

        encoded_text = []

        # encode
        for text in texts:
            encoded = self.enc.encode(text)
            encoded_text.extend(encoded + [self.eos])

        # padding
        for i in range(0, len(encoded_text), block_size):
            chunk = encoded_text[i:i+block_size+1]
            if len(chunk) < block_size:
                chunk.extend([self.eos] * (block_size - len(chunk) + 1))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
        
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]

        x = torch.tensor(chunk[:-1], dtype = torch.long)
        y = torch.tensor(chunk[1:], dtype = torch.long)
        return x, y

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
text = load_dataset("Locutusque/Mind-Corpus")
all_values = []
for conv in text['train']['conversations']:
    for turn in conv:
        all_values.append(turn['value'])

train_dataset = myDataset(all_values, LMconfig) # todo:find a json dataset

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.8,0.2])
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = 8, shuffle = True)

model = GPT(LMconfig)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters of model: {total_params / 1e6} M')

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1000)

def train(model, dataset, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, (x,y) in enumerate(dataset):
        x, y = x.to(device), y.to(device)

        # forward
        logits, loss = model(x, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # step(adjust)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # log
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, batch {batch_idx}, loss: {loss.item()}')
        return total_loss
    
def eval(model, dataset, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            eval_loss += loss.item()

    print(f'Eval loss: {eval_loss}')
    return eval_loss

for epoch in range(2):
    train_loss = train(model, train_loader, optimizer, scheduler, device)
    eval_loss = eval(model, valid_loader, device)

    print(f'Epoch {epoch}, train loss: {train_loss / len(train_loader): 4f}, eval loss: {eval_loss / len(valid_loader): 4f}')

