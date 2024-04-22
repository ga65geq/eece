import pytorch_lightning as pl
from etc.Arguments import add_args, Arguments
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data_prep.data_praperation import DataPrep
from model_trainer.trainer import LitModel
from pytorch_lightning import seed_everything


if __name__ == "__main__":
    import torch.multiprocessing

    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(2023)

    cmd_args = add_args()
    args = Arguments(cmd_args)

    name = args.model_name
    model = LitModel(args)

    if args.use_wandb:
        wandb_logger = WandbLogger(offline=True, name=name, project=args.project_name)
    else:
        wandb_logger = None
    dataprep = DataPrep(args.data_cache_dir, args.dataset)

    train_loader, val_loader, test_loader = dataprep.get_dataloader(num_workers=args.num_workers,
                                                                    batch_size=args.batch_size)

    if args.early_stopping:
        callbacks = [
            EarlyStopping(monitor='val_acc', mode='max', patience=args.patience, min_delta=args.min_delta),
            ModelCheckpoint(save_top_k=args.save_top_k, monitor='val_acc', mode='max', dirpath=args.model_cache_dir,
                            filename=name)
        ]
    else:
        callbacks = [
            ModelCheckpoint(save_top_k=args.save_top_k, monitor='val_acc', mode='max', dirpath=args.model_cache_dir,
                            filename=name)
        ]

    trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', accelerator=args.device, devices=args.device_num,
                         callbacks=callbacks, max_epochs=args.epochs, logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)
