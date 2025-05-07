import lightning as L
import torch
import torch.nn as nn
from utils.losses import KLDLoss
from models import build_network
import numpy as np
from torchmetrics.classification import F1Score


class DANN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        net = build_network(config)

        self.encoder = net.encoder
        self.classifier = net.classifier
        self.discriminator = net.discriminator

        self.config = config

        if config['model']['out_node'] == 1:
            self.f1_score = F1Score(task='binary')
        else:
            self.f1_score = F1Score(task='multiclass', num_classes=4, average="weighted")

        self.binary = (config['model']['out_node'] == 1)
        self.probabilistic = (config['model']['strategy'] != 'Deterministic')

        self.domain_criterion = nn.BCELoss()

        if self.binary:
            self.class_criterion = nn.BCELoss()
        else:
            self.class_criterion = nn.NLLLoss()

        if self.probabilistic:
            self.kldloss = KLDLoss(config['model']['strategy'])

    def training_step(self, batch, batch_idx):
        opt_e, opt_c, opt_d = self.optimizers()

        p = self.get_p()
        lambda_p = self.get_lambda_p(p)

        (src_seq, src_class_label), (tgt_seq, _) = batch

        src_domain_label = torch.zeros(len(src_class_label), dtype=torch.float, device=src_class_label.device)
        tgt_domain_label = torch.ones(len(src_class_label), dtype=torch.float, device=src_class_label.device)
        src_and_tgt_domain_label = torch.concat([src_domain_label, tgt_domain_label])

        opt_d.zero_grad()

        src_feat, src_param = self.encoder(src_seq)
        tgt_feat, tgt_param = self.encoder(tgt_seq)

        feature = torch.concat([src_feat, tgt_feat])
        domain_output = self.discriminator(feature.detach()).squeeze()
        domain_loss = self.domain_criterion(domain_output, src_and_tgt_domain_label)

        self.manual_backward(domain_loss)
        opt_d.step()

        opt_e.zero_grad()
        opt_c.zero_grad()

        class_output = self.classifier(feature[:src_seq.shape[0]]).squeeze()
        domain_output = self.discriminator(feature).squeeze()

        class_loss = self.class_criterion(class_output, src_class_label.float() if self.binary else src_class_label.long())
        domain_loss = self.domain_criterion(domain_output, src_and_tgt_domain_label)

        loss = class_loss - lambda_p * self.config['train']['lambda'] * domain_loss

        if self.probabilistic:
            loss += self.config['train']['nu'][self.config['model']['strategy']] * (
                    self.kldloss(src_param[0], src_param[1]) + self.kldloss(tgt_param[0], tgt_param[1])
            )

        self.manual_backward(loss)
        opt_e.step()
        opt_c.step()

        pred = (class_output > 0.5).long() if self.binary else class_output.argmax(dim=1)
        f1 = self.f1_score(src_class_label, pred)

        self.log('train_class_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', f1, on_step=False, on_epoch=True, prog_bar=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        (src_seq, src_label), (tgt_seq, tgt_label) = batch
        src_feat, src_param = self.encoder(src_seq)
        src_out = self.classifier(src_feat).squeeze()

        src_class_loss = self.class_criterion(src_out, src_label.float() if self.binary else src_label.long())

        src_pred = (src_out > 0.5).long() if self.binary else src_out.argmax(dim=1)
        src_f1 = self.f1_score(src_label, src_pred)

        tgt_feat, tgt_param = self.encoder(tgt_seq)
        tgt_out = self.classifier(tgt_feat).squeeze()

        tgt_class_loss = self.class_criterion(tgt_out, tgt_label.float() if self.binary else tgt_label.long())

        tgt_pred = (tgt_out > 0.5).long() if self.binary else tgt_out.argmax(dim=1)
        tgt_f1 = self.f1_score(tgt_label, tgt_pred)

        self.log('val_src_class_loss', src_class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_src_acc', src_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_class_loss', tgt_class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_tgt_acc', tgt_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        encoder = self.encoder
        classifier = self.classifier
        discriminator = self.discriminator

        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=self.config['train']['encoder_lr'])
        optimizer_c = torch.optim.Adam(classifier.parameters(), lr=self.config['train']['classifier_lr'])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.config['train']['discriminator_lr'])
        return [optimizer_e, optimizer_c, optimizer_d]

    def get_p(self):
        return float(self.global_step / (self.config['train']['num_epochs'] * self.trainer.num_training_batches))

    def get_lambda_p(self, p):
        gamma = 10
        lambda_p = 2. / (1. + np.exp(-gamma * p)) - 1
        return lambda_p

