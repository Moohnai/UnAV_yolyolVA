# python imports
import argparse
import datetime
import glob
import os
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# for visualization
from torch.utils.tensorboard import SummaryWriter

import wandb

# our code
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_multimodal_meta_arch
from libs.utils import (
    ANETdetection,
    ModelEma,
    debugger_is_active,
    fix_random_seed,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
    valid_one_epoch,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    single_or_multi_gpu = "single" if len(cfg["devices"]) == 1 else "multi"

    model_name = f"{single_or_multi_gpu}_gpu_{cfg['opt']['epochs']}_epochs_inter_{cfg['model']['inter_contr_weight']}_intra_{cfg['model']['intra_contr_weight']}_score_v_{cfg['model']['score_V_weight']}_score_a_{cfg['model']['score_A_weight']}_batch_{cfg['loader']['batch_size']}"

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        # ckpt_folder = os.path.join(
        #     cfg['output_folder'], cfg_filename + '_' + str(ts))
        ckpt_folder = os.path.join(cfg["output_folder"], model_name + "_" + str(ts))
    else:
        # ckpt_folder = os.path.join(
        #     cfg['output_folder'], cfg_filename + '_' + str(args.output))
        ckpt_folder = os.path.join(
            cfg["output_folder"], model_name + "_" + str(args.output)
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg["opt"]["learning_rate"] *= len(cfg["devices"])
    cfg["loader"]["num_workers"] *= len(cfg["devices"])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    # update cfg based on dataset attributes
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg["loader"], **cfg["dataset"]
    )

    if cfg["train_cfg"]["evaluate"]:
        val_dataset = make_dataset(
            cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
        )
        val_loader = make_data_loader(
            val_dataset, False, None, **cfg["loader"], **cfg["dataset"]
        )

        # set up evaluator
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            model_name,
            tiou_thresholds=val_db_vars["tiou_thresholds"],
        )

    """3. create model, optimizer, and scheduler"""
    model = make_multimodal_meta_arch(cfg["model_name"], **cfg["model"])

    device = torch.device(cfg["devices"][0] if torch.cuda.is_available() else "cpu")

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg["devices"])

    model.to(device)

    ################
    # ckpt = '/home/mona/UnAV_alignment_Contrastive_yolyolVA/ckpt/single_gpu_40_epochs_inter_0.001_intra_1.0_score_v_0.001_score_a_0.001_batch_8_2024-12-01 20:55:54/model_best.pth.tar'
    # if ".pth.tar" in ckpt:
    #     assert os.path.isfile(ckpt), "CKPT file does not exist!"
    #     ckpt_file = ckpt
    # else:
    #     assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
    #     ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
    #     ckpt_file = ckpt_file_list[-1]
    # checkpoint = torch.load(
    #     ckpt_file,
    #     map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    # )
    # # load ema model instead
    # print("Loading from EMA model ...")
    # model.load_state_dict(checkpoint['state_dict_ema'])
    ################
    # optimizer
    optimizer = make_optimizer(model, cfg["opt"])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(cfg["devices"][0]),
            )
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{:s}' (epoch {:d}".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    ###m:
    # initialize wandb
    if not debugger_is_active():
        wandb.init(
            project="DEL_UnAV",
            group="training_alignment_contrastive_yolyolVA",
            name=model_name,
            config=args,
        )
    ###

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg["model_name"]))

    # start training
    max_epochs = cfg["opt"].get(
        "early_stop_epochs", cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"]
    )
    best_mAP = 0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_stats = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg["train_cfg"]["clip_grad_l2norm"],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )

        # evaluate on val set
        if (epoch + 1) % cfg["train_cfg"]["eval_freq"] == 0 or epoch == max_epochs - 1:
            if cfg["train_cfg"]["evaluate"]:
                print("\nStart evaluating model {:s} ...".format(cfg["model_name"]))
                start = time.time()
                avg_mAP, val_stats = valid_one_epoch(
                    val_loader,
                    model,
                    epoch,
                    evaluator=det_eval,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                )
                end = time.time()
                print("evluation done! Total time: {:0.2f} sec".format(end - start))

                if avg_mAP > best_mAP:
                    best_mAP = avg_mAP
                    save_states = {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_states["state_dict_ema"] = model_ema.module.state_dict()
                    save_checkpoint(save_states, True, file_folder=ckpt_folder)

                # log val stats
                wandb_dict = {}
                for key, value in val_stats.items():
                    wandb_dict["val_epoch_" + key] = value
                wandb.log(wandb_dict, step=epoch)

        ###m:
        # wandb log train stats
        wandb_dict = {}
        for key, value in train_stats.items():
            wandb_dict["train_epoch_" + key] = value
        wandb.log(wandb_dict, step=epoch)
        ###
        # save ckpt once in a while
        if (epoch == max_epochs - 1) or (
            (args.ckpt_freq > 0) and (epoch % args.ckpt_freq == 0) and (epoch > 0)
        ):
            save_states = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_states["state_dict_ema"] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name="epoch_{:03d}.pth.tar".format(epoch),
            )

    # load the best model
    print("Loading the best model ...")
    best_ckpt = os.path.join(ckpt_folder, "model_best.pth.tar")
    assert os.path.isfile(best_ckpt), "Best model does not exist!"
    checkpoint = torch.load(
        best_ckpt, map_location=lambda storage, loc: storage.cuda(cfg["devices"][0])
    )
    model.load_state_dict(checkpoint["state_dict"])

    # evaluate on the validation set
    if cfg["train_cfg"]["evaluate"]:
        print("\nStart evaluating model {:s} ...".format(cfg["model_name"]))
        start = time.time()
        avg_mAP = valid_one_epoch(
            val_loader,
            model,
            epoch,
            evaluator=det_eval,
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )
        end = time.time()
        print("evluation done! Total time: {:0.2f} sec".format(end - start))
        print("Best mAP: {:0.4f}".format(best_mAP))

    # wrap up
    tb_writer.close()
    print("All done!")
    return


if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a point-based transformer for action localization"
    )
    parser.add_argument(
        "--config",
        default="/home/mona/UnAV_alignment_Contrastive_yolyolVA/configs/avel_unav100.yaml",
        help="path to a config file",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=200,
        type=int,
        help="print frequency (default: 20 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=20,
        type=int,
        help="checkpoint frequency (default: every 5 epochs)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)
    ###m:
    wandb.finish()
