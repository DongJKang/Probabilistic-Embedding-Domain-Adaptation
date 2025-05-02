from methods.trainers import (
    train_source_only,
    train_coral,
    train_mmd,
    train_dann,
    train_mcd,
    train_adda,
    train_hhd
)

def train_method(net, source_net, data, config):
    method = config['train']['method']

    if method == 'sourceonly':
        return train_source_only(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            nu=config['train']['nu']
        )

    elif method == 'coral':
        return train_coral(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            coral_lambda=config['train']['coral_lambda'],
            nu=config['train']['nu']
        )

    elif method == 'mmd':
        return train_mmd(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            mmd_lambda=config['train']['mmd_lambda'],
            nu=config['train']['nu']
        )

    elif method == 'dann':
        return train_dann(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            discriminator_lr=config['train']['discriminator_lr'],
            dann_lambda=config['train']['dann_lambda'],
            nu=config['train']['nu']
        )

    elif method == 'mcd':
        return train_mcd(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            num_b_steps=config['train']['mcd_num_b'],
            num_c_steps=config['train']['mcd_num_c'],
            nu=config['train']['nu']
        )

    elif method == 'adda':
        return train_adda(
            net=net,
            source_net=source_net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            discriminator_lr=config['train']['discriminator_lr'],
            num_dis_steps=config['train']['adda_dis_steps'],
            num_enc_steps=config['train']['adda_tgt_steps'],
            nu=config['train']['nu']
        )

    elif method == 'hhd':
        return train_hhd(
            net=net,
            data=data,
            random_seed=config['dataset']['random_seed'],
            num_epochs=config['train']['num_epochs'],
            encoder_lr=config['train']['encoder_lr'],
            classifier_lr=config['train']['classifier_lr'],
            hhd_lambda=config['train']['hhd_lambda'],
            nu=config['train']['nu']
        )

    else:
        raise ValueError(f"Unknown training method: {method}")
