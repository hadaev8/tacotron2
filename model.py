from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths


# https://gist.github.com/CookiePPP/ddbb8a5a9bf18c2e6f79ce3957bd4600
class StepwiseMonotonicAttention(nn.Module):
    def __init__(self, query_dim, value_dim, attention_dim,
                 sigmoid_noise=2.0, score_bias_init=3.5,
                 use_hard_attention=False, attention_score_bias=-4.0):
        super(StepwiseMonotonicAttention, self).__init__()
        self.query_layer = LinearNorm(
            query_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(
            value_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = nn.utils.weight_norm(
            nn.Linear(attention_dim, 1, bias=True), name='weight')
        self.v.bias.data.fill_(score_bias_init)
        self.use_hard_attention = use_hard_attention
        self.score_mask_value = 0.0
        self.sigmoid_noise = sigmoid_noise
        self.attention_score_bias = nn.Parameter(
            torch.Tensor([attention_score_bias]))

    def set_soft_attention(self):
        self.use_hard_attention = False

    def set_hard_attention(self):
        self.use_hard_attention = True

    def monotonic_stepwise_attention(self, p_choose_i, previous_attention, hard_attention):
        # p_choose_i, previous_alignments, previous_score: [B, memory_size]
        # p_choose_i: probability to keep attended to the last attended entry i
        if hard_attention:
            # Given that previous_alignments is one_hot
            move_next_mask = F.pad(previous_attention[:, :-1], (1, 0))
            # [B, memory_size] -> [B]
            stay_prob = torch.sum(p_choose_i * previous_attention, dim=1)
            attention = torch.where(
                stay_prob > 0.5, previous_attention, move_next_mask)
        else:
            attention = previous_attention * p_choose_i + \
                F.pad(previous_attention[:, : -1] *
                      (1.0 - p_choose_i[:, : -1]), (1, 0))
        return attention

    def _stepwise_monotonic_probability_fn(self, score, previous_alignments, sigmoid_noise, hard_attention):
        """
        score: [B, enc_T]
        previous_alignments: [B, enc_T]
        """
        if sigmoid_noise > 0:
            noise = torch.randn(
                score.shape, device=score.device, dtype=score.dtype)
            score = score + sigmoid_noise * noise
        if hard_attention:
            # When mode is hard, use a hard sigmoid
            p_choose_i = (score > 0.).to(score.dtype)
        else:
            p_choose_i = score.sigmoid()
        alignments = self.monotonic_stepwise_attention(
            p_choose_i, previous_alignments, hard_attention)
        return alignments

    def get_alignment_energies(self, query, processed_memory, previous_alignments):
        processed_query = self.query_layer(query).unsqueeze(
            1).expand(-1, processed_memory.size(1), -1)

        score = self.attention_score_bias + \
            self.v(torch.tanh(processed_query + processed_memory)).squeeze(-1)

        alignments = self._stepwise_monotonic_probability_fn(
            score, previous_alignments, self.sigmoid_noise, self.use_hard_attention)

        return alignments

    def forward(self, query, memory, processed_memory, previous_alignments, mask):
        alignment = self.get_alignment_energies(
            query, processed_memory, previous_alignments)

        if mask is not None:
            alignment.data.masked_fill_(
                mask, self.score_mask_value)

        attention_context = torch.bmm(
            alignment.unsqueeze(1), memory).squeeze(1)

        return attention_context, alignment


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int(
                                 (hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim_hidden, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = StepwiseMonotonicAttention(
            query_dim=hparams.attention_rnn_dim, value_dim=hparams.encoder_embedding_dim,
            attention_dim=hparams.attention_dim, sigmoid_noise=hparams.sigmoid_noise)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.attention_hidden_init = nn.Parameter(
            torch.zeros((1, self.attention_rnn_dim)))
        self.attention_cell_init = nn.Parameter(
            torch.zeros((1, self.attention_rnn_dim)))

        self.decoder_hidden_init = nn.Parameter(
            torch.zeros((1, self.decoder_rnn_dim)))
        self.decoder_cell_init = nn.Parameter(
            torch.zeros((1, self.decoder_rnn_dim)))

        self.attention_weights_init = nn.Parameter(torch.ones((1, 1)))
        self.attention_context_init = nn.Parameter(
            torch.zeros((1, self.encoder_embedding_dim)))

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = self.attention_hidden_init.expand(B, -1)
        self.attention_cell = self.attention_cell_init.expand(B, -1)

        self.decoder_hidden = self.decoder_hidden_init.expand(B, -1)
        self.decoder_cell = self.decoder_cell_init.expand(B, -1)

        self.attention_weights = self.attention_weights_init.expand(
            B, MAX_TIME)
        self.attention_context = self.attention_context_init.expand(B, -1)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            self.attention_weights, self.mask)

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        return decoder_hidden_attention_context, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        decoder_hidden_attention_contexts, gate_outputs, alignments = [], [], []
        while len(decoder_hidden_attention_contexts) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(
                decoder_hidden_attention_contexts)]
            decoder_hidden_attention_context, attention_weights = self.decode(
                decoder_input)
            decoder_hidden_attention_contexts += [
                decoder_hidden_attention_context]
            alignments += [attention_weights]

        decoder_hidden_attention_context = torch.stack(
            decoder_hidden_attention_contexts, axis=1)

        mel_outputs = self.linear_projection(decoder_hidden_attention_context)

        gate_outputs = self.gate_layer(
            decoder_hidden_attention_context).squeeze(-1)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            decoder_hidden_attention_context, alignment = self.decode(
                decoder_input)

            mel_output = self.linear_projection(
                decoder_hidden_attention_context)

            gate_output = self.gate_layer(
                decoder_hidden_attention_context).squeeze(-1)
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output
        mel_outputs = torch.stack(mel_outputs, axis=1)
        gate_outputs = torch.stack(gate_outputs, axis=1)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class CTCEsitmator(nn.Module):
    def __init__(self, vocab_size, inp_dim):
        super(CTCEsitmator, self).__init__()

        self.proj = LinearNorm(
            inp_dim, vocab_size + 1, w_init_gain='linear')

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.proj(x).log_softmax(
            dim=-1).transpose(1, 0)

        return x


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.ctc_estimator = CTCEsitmator(
            hparams.n_symbols, hparams.n_mel_channels)
        self.use_ctc = True

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if self.use_ctc:
            ctc_out = self.ctc_estimator(mel_outputs)
        else:
            ctc_out = None

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, ctc_out],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
