import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from utils.metrics import set_all_seeds
from utils.losses import KLDLoss, CORALLoss, MMDLoss, RBF

# ============================================
# Train Source Only
# ============================================
def train_source_only(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier = net[1].to(DEVICE)

    binary = classifier.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()
    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier.train()

        epoch_loss = 0
        epoch_acc = 0

        for seq, label in src_train_loader:
            seq = seq.to(DEVICE).float()
            label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

            optim_e.zero_grad()
            optim_c.zero_grad()

            feature, parameter = encoder(seq)
            output = classifier(feature).squeeze()

            loss = class_criterion(output, label)
            if probabilistic:
                loss += nu[encoder.strategy] * kldloss(parameter[0], parameter[1])

            loss.backward()
            optim_e.step()
            optim_c.step()

            pred = (output > 0.5).long() if binary else output.argmax(dim=1)
            f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / len(src_train_loader))
        train_acc_list.append(epoch_acc / len(src_train_loader))

        # ========== Validate (Source) ==========
        encoder.eval()
        classifier.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        # ========== Validate (Target) ==========
        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[SourceOnly][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')  # Endline after final epoch

    return [encoder, classifier], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_coral(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, coral_lambda, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier = net[1].to(DEVICE)

    binary = classifier.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()
    domain_criterion = CORALLoss().to(DEVICE)
    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier.train()

        epoch_loss = 0
        epoch_acc = 0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        n_iter = min(len(src_iter), len(tgt_iter))

        for _ in range(n_iter):
            src_seq, src_label = next(src_iter)
            tgt_seq, _ = next(tgt_iter)

            src_seq = src_seq.to(DEVICE).float()
            tgt_seq = tgt_seq.to(DEVICE).float()
            src_label = src_label.to(DEVICE).float() if binary else src_label.to(DEVICE).long()

            optim_e.zero_grad()
            optim_c.zero_grad()

            src_feature, src_param = encoder(src_seq)
            tgt_feature, tgt_param = encoder(tgt_seq)

            src_output = classifier(src_feature).squeeze()
            class_loss = class_criterion(src_output, src_label)
            domain_loss = domain_criterion(src_feature, tgt_feature)

            loss = class_loss + coral_lambda * domain_loss
            if probabilistic:
                loss += nu[encoder.strategy] * (kldloss(src_param[0], src_param[1]) + kldloss(tgt_param[0], tgt_param[1]))

            loss.backward()
            optim_e.step()
            optim_c.step()

            pred = (src_output > 0.5).long() if binary else src_output.argmax(dim=1)
            f1 = f1_score(src_label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(epoch_acc / n_iter)

        # ========== Validation ==========
        encoder.eval()
        classifier.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[CORAL][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [encoder, classifier], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_mmd(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, mmd_lambda, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier = net[1].to(DEVICE)

    binary = classifier.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()
    domain_criterion = MMDLoss(kernel=RBF().to(DEVICE)).to(DEVICE)

    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier.train()

        epoch_loss = 0
        epoch_acc = 0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        n_iter = min(len(src_iter), len(tgt_iter))

        for _ in range(n_iter):
            src_seq, src_label = next(src_iter)
            tgt_seq, _ = next(tgt_iter)

            src_seq = src_seq.to(DEVICE).float()
            tgt_seq = tgt_seq.to(DEVICE).float()
            src_label = src_label.to(DEVICE).float() if binary else src_label.to(DEVICE).long()

            optim_e.zero_grad()
            optim_c.zero_grad()

            src_feature, src_param = encoder(src_seq)
            tgt_feature, tgt_param = encoder(tgt_seq)

            src_output = classifier(src_feature).squeeze()

            class_loss = class_criterion(src_output, src_label)
            domain_loss = domain_criterion(src_feature, tgt_feature)

            loss = class_loss + mmd_lambda * domain_loss
            if probabilistic:
                loss += nu[encoder.strategy] * (kldloss(src_param[0], src_param[1]) + kldloss(tgt_param[0], tgt_param[1]))

            loss.backward()
            optim_e.step()
            optim_c.step()

            pred = (src_output > 0.5).long() if binary else src_output.argmax(dim=1)
            f1 = f1_score(src_label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(epoch_acc / n_iter)

        # ========== Validation ==========
        encoder.eval()
        classifier.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[MMD][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [encoder, classifier], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_dann(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, discriminator_lr, dann_lambda, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier = net[1].to(DEVICE)
    discriminator = net[2].to(DEVICE)

    binary = classifier.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()
    domain_criterion = nn.BCELoss()

    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier.train()
        discriminator.train()

        epoch_loss = 0
        epoch_acc = 0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        n_iter = min(len(src_iter), len(tgt_iter))

        for _ in range(n_iter):
            src_seq, src_label = next(src_iter)
            tgt_seq, _ = next(tgt_iter)

            src_seq = src_seq.to(DEVICE).float()
            tgt_seq = tgt_seq.to(DEVICE).float()
            src_label = src_label.to(DEVICE).float() if binary else src_label.to(DEVICE).long()

            # Forward
            src_feature, src_param = encoder(src_seq)
            tgt_feature, tgt_param = encoder(tgt_seq)

            src_output = classifier(src_feature).squeeze()

            # Class loss
            class_loss = class_criterion(src_output, src_label)

            # Domain loss
            src_domain = discriminator(src_feature)
            tgt_domain = discriminator(tgt_feature)

            src_domain_labels = torch.ones(src_domain.size(0), 1).to(DEVICE)
            tgt_domain_labels = torch.zeros(tgt_domain.size(0), 1).to(DEVICE)

            domain_loss = (domain_criterion(src_domain, src_domain_labels) +
                           domain_criterion(tgt_domain, tgt_domain_labels)) / 2

            loss = class_loss + dann_lambda * domain_loss
            if probabilistic:
                loss += nu[encoder.strategy] * (kldloss(src_param[0], src_param[1]) + kldloss(tgt_param[0], tgt_param[1]))

            # Backward
            optim_e.zero_grad()
            optim_c.zero_grad()
            optim_d.zero_grad()

            loss.backward()

            optim_e.step()
            optim_c.step()
            optim_d.step()

            pred = (src_output > 0.5).long() if binary else src_output.argmax(dim=1)
            f1 = f1_score(src_label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(epoch_acc / n_iter)

        # ========== Validation ==========
        encoder.eval()
        classifier.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[DANN][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [encoder, classifier, discriminator], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_mcd(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, num_b_steps, num_c_steps, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier1 = net[1].to(DEVICE)
    classifier2 = net[2].to(DEVICE)

    binary = classifier1.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c1 = torch.optim.Adam(classifier1.parameters(), lr=classifier_lr)
    optim_c2 = torch.optim.Adam(classifier2.parameters(), lr=classifier_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()
    discrepancy_criterion = nn.L1Loss()

    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier1.train()
        classifier2.train()

        epoch_loss = 0
        epoch_acc = 0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        n_iter = min(len(src_iter), len(tgt_iter))

        for _ in range(n_iter):
            src_seq, src_label = next(src_iter)
            tgt_seq, _ = next(tgt_iter)

            src_seq = src_seq.to(DEVICE).float()
            tgt_seq = tgt_seq.to(DEVICE).float()
            src_label = src_label.to(DEVICE).float() if binary else src_label.to(DEVICE).long()

            # Step 1: Train encoder + both classifiers on source
            optim_e.zero_grad()
            optim_c1.zero_grad()
            optim_c2.zero_grad()

            src_feature, src_param = encoder(src_seq)

            out1 = classifier1(src_feature).squeeze()
            out2 = classifier2(src_feature).squeeze()

            loss1 = class_criterion(out1, src_label)
            loss2 = class_criterion(out2, src_label)
            loss = loss1 + loss2

            if probabilistic:
                loss += nu[encoder.strategy] * kldloss(src_param[0], src_param[1])

            loss.backward()
            optim_e.step()
            optim_c1.step()
            optim_c2.step()

            # Step 2: Train classifiers to maximize discrepancy on target
            for _ in range(num_c_steps):
                optim_c1.zero_grad()
                optim_c2.zero_grad()

                tgt_feature, _ = encoder(tgt_seq)
                out1 = classifier1(tgt_feature)
                out2 = classifier2(tgt_feature)

                discrepancy = discrepancy_criterion(out1, out2)
                (-discrepancy).backward()

                optim_c1.step()
                optim_c2.step()

            # Step 3: Train encoder to minimize discrepancy
            for _ in range(num_b_steps):
                optim_e.zero_grad()

                tgt_feature, _ = encoder(tgt_seq)
                out1 = classifier1(tgt_feature)
                out2 = classifier2(tgt_feature)

                discrepancy = discrepancy_criterion(out1, out2)
                discrepancy.backward()

                optim_e.step()

            pred = (out1 > 0.5).long() if binary else out1.argmax(dim=1)
            f1 = f1_score(src_label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(epoch_acc / n_iter)

        # ========== Validation ==========
        encoder.eval()
        classifier1.eval()
        classifier2.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                out1 = classifier1(feature)
                out2 = classifier2(feature)

                output = (out1 + out2) / 2
                output = output.squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                out1 = classifier1(feature)
                out2 = classifier2(feature)

                output = (out1 + out2) / 2
                output = output.squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[MCD][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [encoder, classifier1, classifier2], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_adda(net, source_net, data, random_seed, num_epochs, encoder_lr, discriminator_lr, num_dis_steps, num_enc_steps, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    source_encoder = source_net[0].to(DEVICE)
    classifier = source_net[1].to(DEVICE)
    target_encoder = net[0].to(DEVICE)
    discriminator = net[1].to(DEVICE)

    binary = classifier.out_node == 1
    probabilistic = target_encoder.strategy != 'Deterministic'

    optim_tgt = torch.optim.Adam(target_encoder.parameters(), lr=encoder_lr)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr)

    domain_criterion = nn.BCELoss()
    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()

    if probabilistic:
        kldloss = KLDLoss(target_encoder.strategy)

    # Freeze source encoder
    for param in source_encoder.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        target_encoder.train()
        discriminator.train()

        epoch_loss = 0
        epoch_acc = 0

        tgt_iter = iter(tgt_train_loader)
        n_iter = len(tgt_iter)

        for _ in range(n_iter):
            tgt_seq, _ = next(tgt_iter)
            tgt_seq = tgt_seq.to(DEVICE).float()

            # Step 1: Train Discriminator
            for _ in range(num_dis_steps):
                optim_d.zero_grad()

                with torch.no_grad():
                    src_feature, _ = source_encoder(tgt_seq)

                tgt_feature, tgt_param = target_encoder(tgt_seq)

                src_domain = discriminator(src_feature)
                tgt_domain = discriminator(tgt_feature)

                src_labels = torch.ones(src_domain.size(0), 1).to(DEVICE)
                tgt_labels = torch.zeros(tgt_domain.size(0), 1).to(DEVICE)

                d_loss = (domain_criterion(src_domain, src_labels) + domain_criterion(tgt_domain, tgt_labels)) / 2
                d_loss.backward()
                optim_d.step()

            # Step 2: Train Target Encoder
            for _ in range(num_enc_steps):
                optim_tgt.zero_grad()

                tgt_feature, tgt_param = target_encoder(tgt_seq)
                tgt_domain = discriminator(tgt_feature)

                src_labels = torch.ones(tgt_domain.size(0), 1).to(DEVICE)  # fool the discriminator

                e_loss = domain_criterion(tgt_domain, src_labels)

                if probabilistic:
                    e_loss += nu[target_encoder.strategy] * kldloss(tgt_param[0], tgt_param[1])

                e_loss.backward()
                optim_tgt.step()

            epoch_loss += (d_loss + e_loss).item()

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(0.0)  # No classification training in ADDA

        # ========== Validation ==========
        target_encoder.eval()
        classifier.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = source_encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = target_encoder(seq)
                output = classifier(feature).squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[ADDA][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [target_encoder, discriminator], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]


def train_hhd(net, data, random_seed, num_epochs, encoder_lr, classifier_lr, hhd_lambda, nu):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(random_seed)

    src_train_loader = data['src_train']
    src_val_loader = data['src_val']
    tgt_train_loader = data['tgt_train']
    tgt_val_loader = data['tgt_val']

    encoder = net[0].to(DEVICE)
    classifier1 = net[1].to(DEVICE)
    classifier2 = net[2].to(DEVICE)

    binary = classifier1.out_node == 1
    probabilistic = encoder.strategy != 'Deterministic'

    optim_e = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    optim_c1 = torch.optim.Adam(classifier1.parameters(), lr=classifier_lr)
    optim_c2 = torch.optim.Adam(classifier2.parameters(), lr=classifier_lr)

    class_criterion = nn.BCELoss() if binary else nn.NLLLoss()

    if probabilistic:
        kldloss = KLDLoss(encoder.strategy)

    train_loss_list = []
    train_acc_list = []
    src_val_loss_list = []
    tgt_val_loss_list = []
    src_val_acc_list = []
    tgt_val_acc_list = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # ========== Train ==========
        encoder.train()
        classifier1.train()
        classifier2.train()

        epoch_loss = 0
        epoch_acc = 0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        n_iter = min(len(src_iter), len(tgt_iter))

        for _ in range(n_iter):
            src_seq, src_label = next(src_iter)
            tgt_seq, _ = next(tgt_iter)

            src_seq = src_seq.to(DEVICE).float()
            tgt_seq = tgt_seq.to(DEVICE).float()
            src_label = src_label.to(DEVICE).float() if binary else src_label.to(DEVICE).long()

            optim_e.zero_grad()
            optim_c1.zero_grad()
            optim_c2.zero_grad()

            src_feature, src_param = encoder(src_seq)
            tgt_feature, tgt_param = encoder(tgt_seq)

            src_out1 = classifier1(src_feature).squeeze()
            src_out2 = classifier2(src_feature).squeeze()

            tgt_out1 = classifier1(tgt_feature).squeeze()
            tgt_out2 = classifier2(tgt_feature).squeeze()

            class_loss1 = class_criterion(src_out1, src_label)
            class_loss2 = class_criterion(src_out2, src_label)

            consistency_loss = torch.mean(torch.abs(tgt_out1 - tgt_out2))

            loss = class_loss1 + class_loss2 + hhd_lambda * consistency_loss

            if probabilistic:
                loss += nu[encoder.strategy] * (kldloss(src_param[0], src_param[1]) + kldloss(tgt_param[0], tgt_param[1]))

            loss.backward()
            optim_e.step()
            optim_c1.step()
            optim_c2.step()

            pred = (src_out1 > 0.5).long() if binary else src_out1.argmax(dim=1)
            f1 = f1_score(src_label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

            epoch_loss += loss.item()
            epoch_acc += f1

        train_loss_list.append(epoch_loss / n_iter)
        train_acc_list.append(epoch_acc / n_iter)

        # ========== Validation ==========
        encoder.eval()
        classifier1.eval()
        classifier2.eval()

        src_val_loss = 0
        src_val_acc = 0
        with torch.no_grad():
            for seq, label in src_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                out1 = classifier1(feature)
                out2 = classifier2(feature)

                output = (out1 + out2) / 2
                output = output.squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                src_val_loss += loss.item()
                src_val_acc += f1

        src_val_loss_list.append(src_val_loss / len(src_val_loader))
        src_val_acc_list.append(src_val_acc / len(src_val_loader))

        tgt_val_loss = 0
        tgt_val_acc = 0
        with torch.no_grad():
            for seq, label in tgt_val_loader:
                seq = seq.to(DEVICE).float()
                label = label.to(DEVICE).float() if binary else label.to(DEVICE).long()

                feature, _ = encoder(seq)
                out1 = classifier1(feature)
                out2 = classifier2(feature)

                output = (out1 + out2) / 2
                output = output.squeeze()

                loss = class_criterion(output, label)
                pred = (output > 0.5).long() if binary else output.argmax(dim=1)
                f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0, average='weighted' if not binary else 'binary')

                tgt_val_loss += loss.item()
                tgt_val_acc += f1

        tgt_val_loss_list.append(tgt_val_loss / len(tgt_val_loader))
        tgt_val_acc_list.append(tgt_val_acc / len(tgt_val_loader))

        elapsed_min = (time.time() - start_time) / 60
        print(f'\r[HHD][Epoch {epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f} | '
              f'Src Val Loss: {src_val_loss_list[-1]:.4f} Acc: {src_val_acc_list[-1]:.4f} | '
              f'Tgt Val Loss: {tgt_val_loss_list[-1]:.4f} Acc: {tgt_val_acc_list[-1]:.4f} | '
              f'Time: {elapsed_min:.2f} min', end='')

    print('')
    return [encoder, classifier1, classifier2], [train_loss_list, src_val_loss_list, tgt_val_loss_list], [train_acc_list, src_val_acc_list, tgt_val_acc_list]
