import os
import time
import torch
import random
import logging
import argparse
import datetime
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from statistics import mean
from model.model import MyModel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from model.utils import train_epoch, evaluate
from utils import __balance_val_split, logger
from model.gaussian_noise import GaussianNoise
from datasets.czech_slr_dataset import CzechSLRDataset


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="WLASL_spoter",
        help="Name of the experiment after which the logs and plots will be named",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=100,
        help="Number of classes to be recognized by the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Number of batch size"
    )
    parser.add_argument("--num_worker", type=int, default=0, help="Number of workers")
    parser.add_argument(
        "--num_seq_elements",
        type=int,
        default=108,  # [21(hand)*2 +12(body) ]*2
        help="Hidden dimension of the underlying Transformer model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=379,
        help="Seed with which to initialize all the random components of the training",
    )

    # Data
    parser.add_argument(
        "--training_set_path",
        type=str,
        default="",
        help="Path to the training dataset CSV file",
    )
    parser.add_argument(
        "--testing_set_path",
        type=str,
        default="",
        help="Path to the testing dataset CSV file",
    )
    parser.add_argument(
        "--experimental_train_split",
        type=float,
        default=None,
        help="Determines how big a portion of the training set should be employed (intended for the "
        "gradually enlarging training set experiment from the paper)",
    )

    parser.add_argument(
        "--validation_set",
        type=str,
        choices=["from-file", "split-from-train", "none"],
        default="none",
        help="Type of validation set construction. See README for further rederence",
    )
    parser.add_argument(
        "--validation_set_size",
        type=float,
        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
        " to 'split-from-train'",
    )
    parser.add_argument(
        "--validation_set_path",
        type=str,
        default="",
        help="Path to the validation dataset CSV file",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the model training"
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=1,
        help="Log frequency (frequency of printing all the training info)",
    )

    # Checkpointing
    parser.add_argument(
        "--save_checkpoints",
        type=bool,
        default=True,
        help="Determines whether to save weights checkpoints",
    )

    # Scheduler
    parser.add_argument(
        "--scheduler_factor",
        type=int,
        default=0.1,
        help="Factor for the ReduceLROnPlateau scheduler",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for the ReduceLROnPlateau scheduler",
    )

    # Gaussian noise normalization
    parser.add_argument(
        "--gaussian_mean",
        type=int,
        default=0,
        help="Mean parameter for Gaussian noise layer",
    )
    parser.add_argument(
        "--gaussian_std",
        type=int,
        default=0.001,
        help="Standard deviation parameter for Gaussian noise layer",
    )

    # Visualization
    parser.add_argument(
        "--plot_stats",
        type=bool,
        default=True,
        help="Determines whether continuous statistics should be plotted at the end",
    )
    parser.add_argument(
        "--plot_lr",
        type=bool,
        default=True,
        help="Determines whether the LR should be plotted at the end",
    )

    # Training time
    parser.add_argument(
        "--record_training_time",
        type=bool,
        default=False,
        help="Determines whether continuous statistics of training time should be record",
    )

    # Model settings
    parser.add_argument(
        "--attn_type",
        type=str,
        default="prob",
        help="The attention mechanism used by the model",
    )
    parser.add_argument(
        "--num_enc_layers",
        type=int,
        default=3,
        help="Determines the number of encoder layers",
    )
    parser.add_argument(
        "--num_com_layers",
        type=int,
        default=1,
        help="Determines the number of communicating layers",
    )
    parser.add_argument(
        "--num_dec_layers",
        type=int,
        default=2,
        help="Determines the number of decoder layers",
    )
    parser.add_argument("--FIM", type=bool, default=True, help=" ")
    parser.add_argument(
        "--IA_encoder",
        type=bool,
        default=True,
        help="Determines whether input adaptive encoder will be used",
    )
    parser.add_argument(
        "--IA_decoder",
        type=bool,
        default=False,
        help="Determines whether input adaptive decoder will be used",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Determines the patience for earlier exist",
    )

    return parser


def train(args):
    # MARK: TRAINING PREPARATION AND MODULES
    print(f"num_enc_layers: {args.num_enc_layers}")
    print(f"num_dec_layers: {args.num_dec_layers}, pat_dec: {args.patience}")
    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                args.experiment_name
                + "_"
                + str(args.experimental_train_split).replace(".", "")
                + ".log"
            )
        ],
    )

    # Set device to CUDA only if applicable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    g = torch.Generator()
    g.manual_seed(args.seed)

    transform = transforms.Compose(
        [GaussianNoise(args.gaussian_mean, args.gaussian_std)]
    )
    train_set = CzechSLRDataset(
        args.training_set_path, transform=transform, augmentations=True
    )

    val_loader = None

    train_set, eval_set = __balance_val_split(train_set, 0.2)
    eval_set.transform = None
    eval_set.augmentations = False
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
        num_workers=args.num_worker,
    )

    train_set, val_set = __balance_val_split(train_set, 0.2)

    print(f"train_set: {len(train_set)}")
    print(f"val_set: {len(val_set)}")
    print(f"eval_set: {len(eval_set)}")

    val_set.transform = None
    val_set.augmentations = False
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_worker,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_worker,
    )

    # Construct the model
    slr_model = MyModel(
        num_classes=args.num_classes,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        pat_dec=args.patience,
    )

    # Construct the other modules | Khởi tạo hàm mất mát (loss function)
    cel_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        slr_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(
        parents=True, exist_ok=True
    )
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    # MARK: TRAINING
    slr_model = slr_model.to(device)
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        logger(
            "Starting "
            + args.experiment_name
            + "_"
            + str(args.experimental_train_split).replace(".", "")
            + "..."
        )
    else:
        logger("Starting " + args.experiment_name + "...")

    logger("Training using " + args.training_set_path + "...")

    if args.validation_set == "from-file":
        logger("Validation using " + args.validation_set_path + "...\n\n")

    total_train_time = 0
    avg_train_time_sec_list = []
    # for epoch in range(0):
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, _, _, train_acc, avg_train_time = train_epoch(
            slr_model,
            train_loader,
            cel_criterion,
            optimizer,
            device,
            scheduler=scheduler,
        )
        end_time = time.time()
        train_time = end_time - start_time

        losses.append(train_loss.item() / len(train_loader))
        train_accs.append(train_acc)

        if args.record_training_time:
            avg_train_time_sec_list.append(avg_train_time)
            total_train_time += train_time

        if val_loader:
            pred_correct, pred_correct_topK, pred_all = evaluate(
                slr_model, val_loader, device
            )
            val_acc = pred_correct / pred_all
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(
                    slr_model,
                    "out-checkpoints/"
                    + args.experiment_name
                    + "/checkpoint_t_"
                    + str(checkpoint_index)
                    + ".pth",
                )

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save(
                    slr_model,
                    "out-checkpoints/"
                    + args.experiment_name
                    + "/checkpoint_v_"
                    + str(checkpoint_index)
                    + ".pth",
                )

                logger(
                    f"Save checkpoint for [{str(epoch + 1)}] as "
                    + "out-checkpoints/"
                    + args.experiment_name
                    + "/checkpoint_v_"
                    + str(checkpoint_index)
                    + ".pth"
                )

        if epoch % args.log_freq == 0:
            logger(
                f"[{epoch + 1}] TRAIN loss: {train_loss.item() / len(train_loader)} | Acc: {train_acc}"
            )
            logger(f"[{epoch + 1}] AVG TRAIN time per sample (sec): {avg_train_time} ")

            if val_loader:
                logger("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

                logger(
                    "["
                    + str(epoch + 1)
                    + "] VALIDATION  Top 5 acc: "
                    + str(top_val_acc)
                )

            logger("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(optimizer.param_groups[0]["lr"])

    if args.record_training_time:
        print(
            f"Total training time taken over {args.epochs} epochs: {str(datetime.timedelta(seconds=total_train_time))}"
        )
        print(
            f"Average training time per sample: {str(mean(avg_train_time_sec_list[1:]))}"
        )

        logging.info(
            f"Total training time taken over {args.epochs} epochs: {str(datetime.timedelta(seconds=total_train_time))}"
        )
        logging.info(
            f"Average training time per sample: {str(mean(avg_train_time_sec_list[1:]))}"
        )

    # MARK: TESTING
    top_result_top1, top_result_name_top1 = 0, ""
    top_result_topk, top_result_name_topk = 0, ""
    test_accs_t = test_accs_v = []

    if eval_loader:
        logger("\nTesting checkpointed models starting...\n")
        # for i in [5, 4,3, 2, 1]:
        # for i in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                path_to_load = (
                    "out-checkpoints/"
                    + args.experiment_name
                    + "/checkpoint_"
                    + checkpoint_id
                    + "_"
                    + str(i)
                    + ".pth"
                )

                if not os.path.exists(path_to_load):
                    continue

                tested_model = torch.load(path_to_load, weights_only=False).to(device)

                pred_correct, pred_correct_topK, pred_all = evaluate(
                    tested_model, eval_loader, device
                )

                # === Top 1 ===
                eval_acc_top1 = pred_correct / pred_all

                if checkpoint_id == "v":
                    test_accs_v.append(eval_acc_top1)
                else:
                    test_accs_t.append(eval_acc_top1)

                if eval_acc_top1 > top_result_top1:
                    top_result_top1 = eval_acc_top1
                    top_result_name_top1 = (
                        args.experiment_name
                        + "/checkpoint_"
                        + checkpoint_id
                        + "_"
                        + str(i)
                    )

                # === Top K ===
                eval_acc_topk = pred_correct_topK / pred_all

                if eval_acc_topk > top_result_topk:
                    top_result_topk = eval_acc_topk
                    top_result_name_topk = (
                        args.experiment_name
                        + "/checkpoint_"
                        + checkpoint_id
                        + "_"
                        + str(i)
                    )

                logger(
                    f"checkpoint_{checkpoint_id}_{i:<4}  ->  "
                    f"Top 1: {eval_acc_top1:<10} | "
                    f"Top 5: {eval_acc_topk:<10}"
                )

        path_to_load = "out-checkpoints/" + top_result_name_top1 + ".pth"

        logger(
            "\nThe top result was recorded at "
            + str(top_result_top1)
            + " testing accuracy. The best checkpoint is "
            + top_result_name_top1
            + "."
        )
    logger(f"num_enc_layers: {args.num_enc_layers}")
    logger(f"num_dec_layers: {args.num_dec_layers}, pat_dec: {args.patience}")
    logger("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)

    # LSA64 default args
    parser.set_defaults(
        experiment_name="LSA64",
        training_set_path="datasets/LSA64_60fps.csv",
        validation_set="split-from-train",
        # experimental_train_split = 0.8,
        num_classes=64,
        IA_decoder=True,
        num_worker=2,
        num_enc_layers=1,
        num_dec_layers=1,
        patience=1,
        epochs=50,
    )

    args = parser.parse_args()
    train(args)
