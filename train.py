import argparse
import model as M
import transformer_model as T
import dense_squares_model as D
import nnue_dataset
import pytorch_lightning as pl
import features
import os
import sys
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset


def make_data_loaders(
    train_filenames,
    val_filenames,
    feature_set,
    num_workers,
    batch_size,
    filtered,
    random_fen_skipping,
    wld_filtered,
    early_fen_skipping,
    param_index,
    main_device,
    epoch_size,
    val_size,
    return_policy_index,
):
    # Epoch and validation sizes are arbitrary
    features_name = feature_set.name
    train_infinite = nnue_dataset.SparseBatchDataset(
        features_name,
        train_filenames,
        batch_size,
        num_workers=num_workers,
        filtered=filtered,
        random_fen_skipping=random_fen_skipping,
        wld_filtered=wld_filtered,
        early_fen_skipping=early_fen_skipping,
        param_index=param_index,
        device=main_device,
        return_policy_index=return_policy_index,
    )
    val_infinite = nnue_dataset.SparseBatchDataset(
        features_name,
        val_filenames,
        batch_size,
        filtered=filtered,
        random_fen_skipping=random_fen_skipping,
        wld_filtered=wld_filtered,
        early_fen_skipping=early_fen_skipping,
        param_index=param_index,
        device=main_device,
        return_policy_index=return_policy_index,
    )
    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size),
        batch_size=None,
        batch_sampler=None,
    )
    val = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size),
        batch_size=None,
        batch_sampler=None,
    )
    return train, val


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def list_of_bools(arg):
    return list(map(str2bool, arg.split(",")))


def flatten_once(lst):
    return sum(lst, [])


def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument(
        "datasets",
        action="append",
        nargs="+",
        help="Training datasets (.binpack). Interleaved at chunk level if multiple specified. Same data is used for training and validation if not validation data is specified.",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--validation-data",
        type=str,
        action="append",
        nargs="+",
        dest="validation_datasets",
        help="Validation data to use for validation instead of the training data.",
    )
    parser.add_argument(
        "--lambda",
        default=1.0,
        type=float,
        dest="lambda_",
        help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).",
    )
    parser.add_argument(
        "--start-lambda",
        default=None,
        type=float,
        dest="start_lambda",
        help="lambda to use at first epoch.",
    )
    parser.add_argument(
        "--end-lambda",
        default=None,
        type=float,
        dest="end_lambda",
        help="lambda to use at last epoch.",
    )
    parser.add_argument(
        "--gamma",
        default=0.992,
        type=float,
        dest="gamma",
        help="Multiplicative factor applied to the learning rate after every epoch.",
    )
    parser.add_argument("--lr", default=8.75e-4, type=float, dest="lr", help="Initial learning rate.")
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        dest="num_workers",
        help="Number of worker threads to use for data loading. Currently only works well for binpack.",
    )
    parser.add_argument(
        "--batch-size",
        default=-1,
        type=int,
        dest="batch_size",
        help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.",
    )
    parser.add_argument(
        "--threads",
        default=-1,
        type=int,
        dest="threads",
        help="Number of torch threads to use. Default automatic (cores) .",
    )
    parser.add_argument("--seed", default=42, type=int, dest="seed", help="torch seed to use.")
    parser.add_argument(
        "--smart-fen-skipping",
        action="store_true",
        dest="smart_fen_skipping_deprecated",
        help="If enabled positions that are bad training targets will be skipped during loading. Default: True, kept for backwards compatibility. This option is ignored",
    )
    parser.add_argument(
        "--no-smart-fen-skipping",
        action="store_true",
        dest="no_smart_fen_skipping",
        help="If used then no smart fen skipping will be done. By default smart fen skipping is done.",
    )
    parser.add_argument(
        "--no-wld-fen-skipping",
        action="store_true",
        dest="no_wld_fen_skipping",
        help="If used then no wld fen skipping will be done. By default wld fen skipping is done.",
    )
    parser.add_argument(
        "--random-fen-skipping",
        default=3,
        type=int,
        dest="random_fen_skipping",
        help="skip fens randomly on average random_fen_skipping before using one.",
    )
    parser.add_argument(
        "--resume-from-model",
        dest="resume_from_model",
        help="Initializes training using the weights from the given .pt model",
    )
    parser.add_argument(
        "--network-save-period",
        type=int,
        default=20,
        dest="network_save_period",
        help="Number of epochs between network snapshots. None to disable.",
    )
    parser.add_argument(
        "--save-last-network",
        type=str2bool,
        default=True,
        dest="save_last_network",
        help="Whether to always save the last produced network.",
    )
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=100000000,
        dest="epoch_size",
        help="Number of positions per epoch.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000000,
        dest="validation_size",
        help="Number of positions per validation step.",
    )
    parser.add_argument(
        "--param-index",
        type=int,
        default=0,
        dest="param_index",
        help="Indexing for parameter scans.",
    )
    parser.add_argument(
        "--early-fen-skipping",
        type=int,
        default=-1,
        dest="early_fen_skipping",
        help="Skip n plies from the start.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=1,
        dest="n_heads",
        help="Number of heads for the model",
    )
    parser.add_argument(
        "--depth-list",
        type=list_of_ints,
        dest="depth_list",
        help="list of depths for the transformer model. if n depths are specified, n - 1 layers will be used.",
    )
    parser.add_argument(
        "--dff-list",
        type=list_of_ints,
        dest="dff_list",
        help="list of dffs for the transformer model. length of this list must be exactly one less than depth_list.",
    )
    parser.add_argument(
        "--residual-connection-list",
        type=list_of_bools,
        dest="residual_connection_list",
        help="which transformer model layers residuals connections should be enabled. length must be exactly one less than depth_list.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        dest="depth",
        help="depth for DenseSquares model",
    )
    parser.add_argument(
        "--smolgen-initial",
        type=int,
        default=64,
        dest="smolgen_initial",
        help="smolgen initial depth",
    )
    parser.add_argument(
        "--smolgen-hidden",
        type=int,
        default=64,
        dest="smolgen_hidden",
        help="smolgen hidden depth",
    )
    parser.add_argument(
        "--eval-hidden-depth",
        type=int,
        default=128,
        dest="eval_hidden_depth",
        help="eval hidden depth",
    )
    parser.add_argument(
        "--activation-function",
        default="relu",
        dest="activation_function",
        help="activation function for ffns",
    )
    parser.add_argument(
        "--return-policy-index",
        type=str2bool,
        default=False,
        dest="return_policy_index",
        help="policy index will be included in each batch if set to true",
    )
    parser.add_argument(
        "--policy-classification-weight",
        type=float,
        default=0.0001,
        dest="policy_classification_weight",
        help="multiplier for policy classification loss",
    )
    features.add_argparse_args(parser)
    args = parser.parse_args()

    args.datasets = flatten_once(args.datasets)
    if args.validation_datasets:
        args.validation_datasets = flatten_once(args.validation_datasets)
    else:
        args.validation_datasets = []

    for dataset in args.datasets:
        if not os.path.exists(dataset):
            raise Exception("{0} does not exist".format(dataset))

    for val_dataset in args.validation_datasets:
        if not os.path.exists(val_dataset):
            raise Exception("{0} does not exist".format(val_dataset))

    train_datasets = args.datasets
    val_datasets = train_datasets
    if len(args.validation_datasets) > 0:
        val_datasets = args.validation_datasets

    if (args.start_lambda is not None) != (args.end_lambda is not None):
        raise Exception("Either both or none of start_lambda and end_lambda must be specified.")

    feature_set = features.get_feature_set_from_name(args.features)

    start_lambda = args.start_lambda or args.lambda_
    end_lambda = args.end_lambda or args.lambda_
    max_epoch = args.max_epochs or 800

    if args.features == "DenseSquares":
        nnue = D.DenseSquares(
            depth=args.depth,
            eval_hidden=args.eval_hidden_depth,
            start_lambda=start_lambda,
            max_epoch=max_epoch,
            end_lambda=end_lambda,
            activation_function=args.activation_function,
            policy_classification_weight=args.policy_classification_weight,
            gamma=args.gamma,
            lr=args.lr,
        )
    elif args.features == "Transformer":
        nnue = T.Transformer(
            n_heads=args.n_heads,
            depth_list=args.depth_list,
            dff_list=args.dff_list,
            residual_connection_list=args.residual_connection_list,
            start_lambda=start_lambda,
            max_epoch=max_epoch,
            end_lambda=end_lambda,
            smolgen_initial=args.smolgen_initial,
            smolgen_hidden=args.smolgen_hidden,
            eval_hidden=args.eval_hidden_depth,
            activation_function=args.activation_function,
            policy_classification_weight=args.policy_classification_weight,
            gamma=args.gamma,
            lr=args.lr,
        )
        if args.resume_from_model is not None:
            print("loading from", args.resume_from_model)
            nnue = nnue.load_from_checkpoint(
                args.resume_from_model,
                n_heads=args.n_heads,
                depth_list=args.depth_list,
                dff_list=args.dff_list,
                start_lambda=start_lambda,
                max_epoch=max_epoch,
                end_lambda=end_lambda,
                smolgen_initial=args.smolgen_initial,
                smolgen_hidden=args.smolgen_hidden,
                eval_hidden=args.eval_hidden_depth,
                activation_function=args.activation_function,
                policy_classification_weight=args.policy_classification_weight,
                gamma=args.gamma,
                lr=args.lr,
                strict=False,
            )
    else:
        if args.resume_from_model is None:
            nnue = M.NNUE(
                feature_set=feature_set,
                start_lambda=start_lambda,
                max_epoch=max_epoch,
                end_lambda=end_lambda,
                gamma=args.gamma,
                lr=args.lr,
                param_index=args.param_index,
            )
            nnue.set_feature_set(feature_set)
            nnue.start_lambda = start_lambda
            nnue.end_lambda = end_lambda
            nnue.max_epoch = max_epoch
            nnue.gamma = args.gamma
            nnue.lr = args.lr
            nnue.param_index = args.param_index

        else:
            nnue = torch.load(args.resume_from_model)
            nnue.set_feature_set(feature_set)
            nnue.start_lambda = start_lambda
            nnue.end_lambda = end_lambda
            nnue.max_epoch = max_epoch
            # we can set the following here just like that because when resuming
            # from .pt the optimizer is only created after the training is started
            nnue.gamma = args.gamma
            nnue.lr = args.lr
            nnue.param_index = args.param_index

    print("Feature set: {}".format(feature_set.name))
    print("Num real features: {}".format(feature_set.num_real_features))
    print("Num virtual features: {}".format(feature_set.num_virtual_features))
    print("Num features: {}".format(feature_set.num_features))

    print("Training with: {}".format(train_datasets))
    print("Validating with: {}".format(val_datasets))

    pl.seed_everything(args.seed)
    print("Seed {}".format(args.seed))

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 16384
    print("Using batch size {}".format(batch_size))

    print("Smart fen skipping: {}".format(not args.no_smart_fen_skipping))
    print("WLD fen skipping: {}".format(not args.no_wld_fen_skipping))
    print("Random fen skipping: {}".format(args.random_fen_skipping))
    print("Skip early plies: {}".format(args.early_fen_skipping))
    print("Param index: {}".format(args.param_index))

    if args.threads > 0:
        print("limiting torch to {} threads.".format(args.threads))
        t_set_num_threads(args.threads)

    logdir = args.default_root_dir if args.default_root_dir else "logs/"
    print("Using log dir {}".format(logdir), flush=True)

    tb_logger = pl_loggers.TensorBoardLogger(logdir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=args.save_last_network,
        every_n_epochs=args.network_save_period,
        save_top_k=-1,
    )
    from pytorch_lightning.strategies.ddp import DDPStrategy

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=tb_logger, strategy=DDPStrategy(find_unused_parameters=False)
    )

    main_device = (
        trainer.strategy.root_device if trainer.strategy.root_device.index is None else "cuda:" + str(trainer.strategy.root_device.index)
    )

    nnue.to(device=main_device)

    print("Using c++ data loader")
    train, val = make_data_loaders(
        train_datasets,
        val_datasets,
        feature_set,
        args.num_workers,
        batch_size,
        not args.no_smart_fen_skipping,
        args.random_fen_skipping,
        not args.no_wld_fen_skipping,
        args.early_fen_skipping,
        args.param_index,
        main_device,
        args.epoch_size,
        args.validation_size,
        args.return_policy_index,
    )

    trainer.fit(nnue, train, val)

    with open(os.path.join(logdir, "training_finished"), "w"):
        pass


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        os.system(f'wmic process where processid="{os.getpid()}" call terminate >nul')
