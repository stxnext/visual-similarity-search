from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import Any

import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from metrics.consts import (
    DATALOADER_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_EMBEDDER_LAYERS,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SAMPLER_M,
    DEFAULT_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    SIZE,
)
from metrics.nets import get_trunk_embedder
from metrics.utils import (
    DatasetCombined,
    get_transformation_with_size,
    save_training_meta,
)


# TODO: check if it wont be easier to load all data from json, rather that passing it to cli
def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument("--data_dir", type=Path, nargs="?", help="Path for data dir")
    parser.add_argument(
        "--meta", type=Path, nargs="?", help="Path for meta file of dataset"
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="?",
        help="Name of training, used to create logs, models directories",
        default="metric_project",
    )
    parser.add_argument(
        "--trunk_model",
        type=str,
        nargs="?",
        help="Name of pretrained model from torchvision",
        default="resnet50",
    )
    parser.add_argument(
        "--embedder_layers", nargs="+", type=int, default=DEFAULT_EMBEDDER_LAYERS
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs="?",
        help="Train/test split factor",
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="Batch size for training",
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?",
        help="Number of epochs in training",
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--lr", type=float, nargs="?", help="Default learning rate", default=DEFAULT_LR
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        nargs="?",
        help="Weight decay for learning rate",
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--sampler_m",
        type=float,
        nargs="?",
        help="Number of samples per class",
        default=DEFAULT_SAMPLER_M,
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs="?",
        help="Input size (width and height) used for resizing",
        default=SIZE[0],
    )
    return parser.parse_args()


def main() -> Any:
    # parse args and create logging directories
    args = parse_args()
    logs_dir = Path(args.name)
    logs_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()

    # get neural net models and custom dataset for easier manipulation during training setup
    trunk, embedder = get_trunk_embedder(args.trunk_model, args.embedder_layers)
    transformation = get_transformation_with_size(args.input_size)
    dataset = DatasetCombined.get_dataset(
        args.meta,
        args.data_dir,
        split=args.split,
        transformation={"train": transformation, "test": transformation},
    )

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(
        trunk.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    embedder_optimizer = torch.optim.Adam(
        embedder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # TODO: add feature of choosing those params from cli/config
    # set pml specific losses, miners, samplers
    loss = losses.CircleLoss()
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    sampler = samplers.MPerClassSampler(
        dataset.y_train,
        m=args.sampler_m,
        length_before_new_iter=len(dataset.train_dataset),
    )

    # package above stuff into dictionaries compatible with pml
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    dataset_dict = {"val": dataset.test_dataset}

    # create hooks for training
    model_dir = str((logs_dir / "models").absolute())
    csv_dir = str((logs_dir / "logs").absolute())
    tensorboard_dir = str((logs_dir / "training_logs").absolute())

    # set up hooks for saving models and logging
    record_keeper, _, _ = logging_presets.get_record_keeper(csv_dir, tensorboard_dir)
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester and add it to end of epoch hook
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        # visualizer=umap.UMAP(),
        # visualizer_hook=visualizer_hook,
        dataloader_num_workers=DATALOADER_WORKERS,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_dir)

    trainer = trainers.MetricLossOnly(
        models,
        optimizers,
        args.batch_size,
        loss_funcs,
        mining_funcs,
        dataset.train_dataset,
        sampler=sampler,
        dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    pprint(vars(args))
    save_training_meta(vars(args), path=logs_dir)
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
