import lightning as L
import torch
import torch.nn as nn
from utils.losses import KLDLoss
from models import build_network
from torchmetrics.classification import F1Score


class HHD(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        net = build_network(config)

        self.encoder = net.encoder
        self.classifier = net.classifier
        self.second_classifier = net.second_classifier

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
        opt_e, opt_c, opt_c2 = self.optimizers()

        (src_seq, src_class_label), (tgt_seq, _) = batch

        opt_e.zero_grad()
        opt_c.zero_grad()
        opt_c2.zero_grad()

        src_feat, src_param = self.encoder(src_seq)
        tgt_feat, tgt_param = self.encoder(tgt_seq)
        src_out1 = self.classifier(src_feat).squeeze()
        src_out2 = self.second_classifier(src_feat).squeeze()
        tgt_out1 = self.classifier(tgt_feat).squeeze()
        tgt_out2 = self.second_classifier(tgt_feat).squeeze()

        loss_s1 = self.class_criterion(src_out1, src_class_label.float() if self.binary else src_class_label.long())
        loss_s2 = self.class_criterion(src_out2, src_class_label.float() if self.binary else src_class_label.long())
        loss_s = loss_s1 + loss_s2

        loss_div = torch.abs(
            torch.mean(torch.abs(src_out1 - src_out2)) - torch.mean(torch.abs(tgt_out1 - tgt_out2)))  # L1 Norm

        loss = loss_s + self.config['train']['lambda'] * loss_div

        if self.probabilistic:
            loss += self.config['train']['nu'][self.config['model']['strategy']] * (
                    self.kldloss(src_param[0], src_param[1]) + self.kldloss(tgt_param[0], tgt_param[1])
            )

        self.manual_backward(loss)
        opt_e.step()
        opt_c.step()
        opt_c2.step()

        output_ensemble = (src_out1 + src_out2) / 2

        pred = (output_ensemble > 0.5).long() if self.binary else output_ensemble.argmax(dim=1)

        f1 = self.f1_score(src_class_label, pred)

        self.log('train_div_loss', loss_div, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', f1, on_step=False, on_epoch=True, prog_bar=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        (src_seq, src_label), (tgt_seq, tgt_label) = batch
        src_feat, src_param = self.encoder(src_seq)
        src_out1 = self.classifier(src_feat).squeeze()
        src_out2 = self.second_classifier(src_feat).squeeze()

        src_output_ensemble = (src_out1 + src_out2) / 2

        src_class_loss = self.class_criterion(src_output_ensemble, src_label.float() if self.binary else src_label.long())

        src_pred = (src_output_ensemble > 0.5).long() if self.binary else src_output_ensemble.argmax(dim=1)

        src_f1 = self.f1_score(src_label, src_pred)

        tgt_feat, tgt_param = self.encoder(tgt_seq)
        tgt_out1 = self.classifier(tgt_feat).squeeze()
        tgt_out2 = self.second_classifier(tgt_feat).squeeze()

        tgt_output_ensemble = (tgt_out1 + tgt_out2) / 2

        tgt_class_loss = self.class_criterion(tgt_output_ensemble, tgt_label.float() if self.binary else tgt_label.long())

        tgt_pred = (tgt_output_ensemble > 0.5).long() if self.binary else tgt_output_ensemble.argmax(dim=1)

        tgt_f1 = self.f1_score(tgt_label, tgt_pred)

        self.log('val_src_class_loss', src_class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_src_acc', src_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_class_loss', tgt_class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_acc', tgt_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        encoder = self.encoder
        classifier = self.classifier
        second_classifier = self.second_classifier

        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=self.config['train']['encoder_lr'])
        optimizer_c = torch.optim.Adam(classifier.parameters(), lr=self.config['train']['classifier_lr'])
        optimizer_c2 = torch.optim.Adam(second_classifier.parameters(), lr=self.config['train']['classifier_lr'])
        return [optimizer_e, optimizer_c, optimizer_c2]
