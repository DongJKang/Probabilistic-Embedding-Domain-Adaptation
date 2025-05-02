import lightning as L
import torch
import torch.nn as nn
from utils.losses import KLDLoss
from models import build_network
from lightning_modules import SO
from torchmetrics.classification import F1Score


class ADDA(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        net = build_network(config)

        self.encoder = net.encoder
        self.encoder.load_state_dict(SO.load_from_checkpoint(config['model']['ckpt_path']).encoder.state_dict())
        self.classifier = net.classifier
        self.discriminator = net.discriminator

        self.tgt_steps_per_epoch = config['train']['tgt_steps']
        self.dis_steps_per_epoch = config['train']['dis_steps']

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

        self.config = config


    def training_step(self, batch, batch_idx):
        opt_e, opt_d = self.optimizers()

        (src_seq, src_class_label), (tgt_seq, _) = batch

        for _ in range(max(1, self.config['train']['dis_steps'])):
            opt_d.zero_grad()

            src_domain_label = torch.zeros(len(src_class_label), dtype=torch.float, device=src_class_label.device)
            tgt_domain_label = torch.ones(len(src_class_label), dtype=torch.float, device=src_class_label.device)

            src_feat, src_param = self.encoder(src_seq)
            tgt_feat, tgt_param = self.encoder(tgt_seq)

            src_domain_output = self.discriminator(src_feat.detach()).squeeze()
            tgt_domain_output = self.discriminator(tgt_feat.detach()).squeeze()

            src_dis_loss = self.domain_criterion(src_domain_output, src_domain_label)
            tgt_dis_loss = self.domain_criterion(tgt_domain_output, tgt_domain_label)

            dis_loss = src_dis_loss + tgt_dis_loss

            self.manual_backward(dis_loss)
            opt_d.step()

        for _ in range(max(1, self.config['train']['tgt_steps'])):
            opt_e.zero_grad()

            tgt_domain_label = torch.zeros(len(src_class_label), dtype=torch.float, device=src_class_label.device)

            tgt_feat, tgt_param = self.encoder(tgt_seq)
            tgt_domain_output = self.discriminator(tgt_feat).squeeze()
            tgt_loss = self.domain_criterion(tgt_domain_output, tgt_domain_label)

            if self.probabilistic:
                tgt_loss += self.config['train']['nu'][self.config['model']['strategy']] * (
                        self.kldloss(tgt_param[0], tgt_param[1])
                )

            self.manual_backward(tgt_loss)
            opt_e.step()

        self.log('train_dis_loss', dis_loss, on_step=False, on_epoch=True)
        self.log('train_tgt_loss', tgt_loss, on_step=False, on_epoch=True)


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

        self.log('val_src_class_loss', src_class_loss, on_step=False, on_epoch=True)
        self.log('val_src_acc', src_f1, on_step=False, on_epoch=True)
        self.log('val_tgt_class_loss', tgt_class_loss, on_step=False, on_epoch=True)
        self.log('val_tgt_acc', tgt_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        encoder = self.encoder
        discriminator = self.discriminator

        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=self.config['train']['encoder_lr'])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.config['train']['discriminator_lr'])
        return [optimizer_e, optimizer_d]
