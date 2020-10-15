from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, hp):
        super(Tacotron2Loss, self).__init__()
        self.use_ctc = True
        self.ctc_loss = nn.CTCLoss(
            blank=hp.n_symbols, reduction='mean', zero_infinity=True)

    def forward(self, model_output, targets, x):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, ctc_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        if self.use_ctc:
            ctc_loss = 0.1 * self.ctc_loss(ctc_out, x[0], x[4], x[1])
            if ctc_loss < 0.02:
                self.use_ctc = False
            mel_loss = mel_loss + ctc_loss

        return mel_loss + gate_loss
