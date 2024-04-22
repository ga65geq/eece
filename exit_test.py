from data_prep.data_praperation import DataPrep
from etc.Arguments import add_args, Arguments
from model_trainer.trainer import LitModel
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(2023)

    cmd_args = add_args()
    args = Arguments(cmd_args)
    device = torch.device(args.device_name)

    model = LitModel.load_from_checkpoint(args.ckpt_path, args=args)
    model = model.model
    model = model.eval()
    model.to(device)

    dataprep = DataPrep(args.data_cache_dir, args.dataset)

    train_loader, val_loader, test_loader = dataprep.get_dataloader(batch_size=args.batch_size)
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            print(target)
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output[0].argmax(dim=1, keepdim=True)
            print(pred)
            if args.snet:
                model.exit_strategy.clear()
            correct += pred.eq(target.view_as(pred)).sum().item()
        print("accuracy: {}".format(correct / len(test_loader.dataset)))
    model.statistic.print()






