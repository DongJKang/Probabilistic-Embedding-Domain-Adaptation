import lightning as L
import torch
import torch.nn as nn
from utils.losses import KLDLoss
from models import build_network
from torchmetrics.classification import F1Score

class SO(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        net = build_network(config)

        self.encoder = net.encoder
        self.classifier = net.classifier

        self.config = config

        if config['model']['out_node'] == 1:
            self.f1_score = F1Score(task='binary')
        else:
            self.f1_score = F1Score(task='multiclass', num_classes=4, average="weighted")

        self.binary = (config['model']['out_node'] == 1)
        self.probabilistic = (config['model']['strategy'] != 'Deterministic')

        if self.binary:
            self.class_criterion = nn.BCELoss()
        else:
            self.class_criterion = nn.NLLLoss()

        if self.probabilistic:
            self.kldloss = KLDLoss(config['model']['strategy'])

    def training_step(self, batch, batch_idx):
        opt_e, opt_c = self.optimizers()
        opt_e.zero_grad()
        opt_c.zero_grad()

        (src_seq, src_label), _ = batch
        feature, param = self.encoder(src_seq)
        output = self.classifier(feature).squeeze()

        loss = self.class_criterion(output, src_label.float() if self.binary else src_label.long())
        if self.probabilistic:
            loss += self.config['train']['nu'][self.config['model']['strategy']] * self.kldloss(param[0], param[1])

        self.manual_backward(loss)
        opt_e.step()
        opt_c.step()

        pred = (output > 0.5).long() if self.binary else output.argmax(dim=1)
        f1 = self.f1_score(src_label, pred)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (src_seq, src_label), (tgt_seq, tgt_label) = batch
        src_feat, src_param = self.encoder(src_seq)
        src_out = self.classifier(src_feat).squeeze()

        src_loss = self.class_criterion(src_out, src_label.float() if self.binary else src_label.long())
        if self.probabilistic:
            src_loss += self.config['train']['nu'][self.config['model']['strategy']] * self.kldloss(src_param[0], src_param[1])

        src_pred = (src_out > 0.5).long() if self.binary else src_out.argmax(dim=1)
        src_f1 = self.f1_score(src_label, src_pred)

        tgt_feat, tgt_param = self.encoder(tgt_seq)
        tgt_out = self.classifier(tgt_feat).squeeze()

        tgt_loss = self.class_criterion(tgt_out, tgt_label.float() if self.binary else tgt_label.long())
        if self.probabilistic:
            tgt_loss += self.config['train']['nu'][self.config['model']['strategy']] * self.kldloss(tgt_param[0], tgt_param[1])

        tgt_pred = (tgt_out > 0.5).long() if self.binary else tgt_out.argmax(dim=1)
        tgt_f1 = self.f1_score(tgt_label, tgt_pred)

        self.log('val_src_loss', src_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_src_acc', src_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_loss', tgt_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_acc', tgt_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        encoder = self.encoder
        classifier = self.classifier

        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=self.config['train']['encoder_lr'])
        optimizer_c = torch.optim.Adam(classifier.parameters(), lr=self.config['train']['classifier_lr'])
        return [optimizer_e, optimizer_c]

