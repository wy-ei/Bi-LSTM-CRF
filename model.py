import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.conditional_random_field import ConditionalRandomField


class BiLSTM_CRF(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        
        self.name = args.name
        self.hidden_size = args.hidden_size
        self.num_tags = args.num_tags
        self.embedding = nn.Embedding(args.embed_size, args.embed_dim)

        self.crf = ConditionalRandomField(self.num_tags, args.condtraints)
        self.lstm =  nn.LSTM(input_size=args.embed_dim,
                             hidden_size=args.hidden_size // 2,
                             num_layers=1, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.num_tags)

        self.device = args.device
        self.dropout = nn.Dropout(args.dropout)

    def get_logits(self, sequences):
        batch_size = sequences.shape[0]
        sequences = sequences.transpose(0, 1)

        embeded = self.embedding(sequences) # (sequence_len, batch_size, embedding_size)

        h0 = torch.randn(2, batch_size, self.hidden_size // 2, device=sequences.device)
        c0 = torch.randn(2, batch_size, self.hidden_size // 2, device=sequences.device)

        outputs, _ = self.lstm(embeded, (h0, c0))

        outputs = self.dropout(outputs)

        outputs = outputs.transpose(0, 1) # (batch_size, sequence_len, hidden_size)

        logits = self.linear(outputs)

        return logits

    def forward(self, sequences: torch.Tensor, tags: torch.Tensor, mask) -> torch.Tensor:
        logits = self.get_logits(sequences)
        log_likelihood = self.crf(logits, tags, mask)
        loss = -log_likelihood
        return loss

    def predict(self, sequences, mask):
        logits = self.get_logits(sequences)
        best_path = self.crf.viterbi_tags(logits, mask)
        tags_pred = [tags for tags, score in best_path]
        return tags_pred