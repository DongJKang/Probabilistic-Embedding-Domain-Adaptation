import argparse
import yaml
import lightning as L
from lightning_modules import SO, CORAL, MMD, DANN, ADDA, MCD, HHD
from utils.door_datamodule import DoorDataModule
import sys
from lightning.pytorch.loggers import CSVLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config['train']['method'] == 'sourceonly':
        model = SO(config=config)
    elif config['train']['method'] == 'coral':
        model = CORAL(config=config)
    elif config['train']['method'] == 'mmd':
        model = MMD(config=config)
    elif config['train']['method'] == 'dann':
        model = DANN(config=config)
    elif config['train']['method'] == 'adda':
        model = ADDA(config=config)
    elif config['train']['method'] == 'mcd':
        model = MCD(config=config)
    elif config['train']['method'] == 'hhd':
        model = HHD(config=config)
    else:
        raise (
            "Undefined method. Set method to sourceonly, coral, mmd, dann, adda, mcd, or hhd")

    datamodule = DoorDataModule(
        src_str='Lisa_4c',
        tgt_str='Ryan_4c',
        batch_size=32,
        seed=0,
    )

    logger = CSVLogger(save_dir="logs", name=config['train']['method'])

    trainer = L.Trainer(
        max_epochs=config['train']['num_epochs'],
        accelerator='auto',
        devices='auto',
        logger=logger,   # basic Lightning CSVLogger
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
