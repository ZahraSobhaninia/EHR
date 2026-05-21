
import torch
import torch.nn as nn


class ClassifierGRU_AttnMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = True,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_hidden: int = 256,
        attn_heads: int = 1,
    ):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be even for bidirectional GRU"
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.h = hidden_size // 2 if bidirectional else hidden_size

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=self.h,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = self.h * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(self.out_dim)
        self.attn = nn.Linear(self.out_dim, attn_heads)  
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, hidden_states, attention_mask, return_attention=False, **_):
        lengths_cpu = attention_mask.sum(dim=1).to(torch.long).cpu()
        safe_lengths = torch.clamp(lengths_cpu, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            hidden_states, safe_lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B, T, out_dim)

        out = self.norm(out)
        device = out.device

      
        mask = attention_mask
        if (lengths_cpu == 0).any():
            mask = mask.clone()
            mask[lengths_cpu == 0, 0] = 1


        mask = mask.to(out.dtype).unsqueeze(-1).to(device)  # (B, T, 1)

        # attention scores: (B, T, heads)
        attn_scores = self.attn(out)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        attn_weights = torch.softmax(attn_scores, dim=1)   
        attn_weights = self.attn_dropout(attn_weights)     # (B, T, heads)


        context = torch.einsum("bth,btd->bhd", attn_weights, out)  # (B, heads, out_dim)
        x = context.mean(dim=1)  # (B, out_dim)

        logits = self.mlp(x)
        if return_attention:
            return logits, attn_weights.mean(dim=-1).detach().cpu()
        return logits



class BiGRU(torch.nn.Module):
    def __init__(self, hidden_size,num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden_size = hidden_size // 2
        self.rnn = torch.nn.GRU(
            hidden_size, self.rnn_hidden_size, batch_first=True, bidirectional=True
            
        )
        # Add layer normalization
        self.norm = torch.nn.LayerNorm(hidden_size)
        # Adjust the input size of the classifier based on the bidirectionality
        classifier_input_size = hidden_size
        self.classifier = torch.nn.Linear(classifier_input_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )

        output, _ = self.rnn(packed)
        # Unpack it back to a padded sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_sequence_idx = lengths - 1

        # Use the last output of the RNN as input to the classifier
        # When bidirectional, we need to concatenate the last output from the forward
        # pass and the first output from the backward pass
        forward_output = output[
            torch.arange(output.shape[0]), last_sequence_idx, : self.rnn_hidden_size
        ]  # Last non-padded output from the forward pass
        backward_output = output[
            :, 0, self.rnn_hidden_size :
        ]  # First output from the backward pass
        x = torch.cat((forward_output, backward_output), dim=-1)
        # Apply layer normalization
        x = self.norm(x)
        if return_embedding:
            return x
        x = self.classifier(x)
        return x

class FineTuneHead(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pool = ClassifierGRU_AttnMLP(hidden_size)

    def forward(self, hidden_states, attention_mask, return_attention=False, **_):
        return self.pool(hidden_states, attention_mask, return_attention=return_attention)