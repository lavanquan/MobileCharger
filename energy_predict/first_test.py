import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim: int,emb_dim: int, hid_dim: int, n_layers: int, dropout_prop: float):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.drop_out_prop = dropout_prop

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout_prop)
        self.dropout = nn.Dropout(dropout_prop)

    def forward(self, input):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [src sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout_prob: float):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn  = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout_prob)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputP, hidden, cell):
        # input = [batch size]
        # hidden = [ n layers * n directions, batch size, hid dim]
        # cell = [ n layers * n directions, batch size, hid dim]

        # n directions in the decder will both always be 1, therefore:
        # hidden = [ n layers, batch size, hid dim]
        # context = [ n layers, batch size, hid dim]

        inputP = inputP.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal"
        assert encoder.n_layers == encoder.n_layers, "Encoder and decoder must have equal number of layers"

    def forward(self, src, trg, teacher_force_ratio=0.5):

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_force_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        return outputs












