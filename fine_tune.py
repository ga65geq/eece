from etc.Arguments import add_args, Arguments
from searcher.threshhold_searcher import Searcher
from pytorch_lightning import seed_everything


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(2023)

    cmd_args = add_args()
    args = Arguments(cmd_args)

    searcher = Searcher(args)
    print(searcher.search())