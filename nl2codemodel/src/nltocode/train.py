import argparse
import logging as log
from datetime import datetime
from multiprocessing import freeze_support

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from nltocode.datamodule import NL2CodeTrainDataModule
from nltocode.nl2code import NL2CodeSystem


def get_args():
    parser = argparse.ArgumentParser("Train model", fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument("--train-valid-data-path", type=str, default=None)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--valid-data-path", type=str, default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.2)

    # Data preprocessing parameters
    parser.add_argument("--vocabsrc-size", type=int, default=800)
    parser.add_argument("--vocabtgt-size", type=int, default=800)
    parser.add_argument("--vocabchar-size", type=int, default=800)

    # parser.add_argument("--vocab-pad-id", type=int, default=0) # Deactivated as implicitly assumed to be 0 in code
    parser.add_argument("--max-src-sentence-length", type=int, default=1000)
    parser.add_argument("--max-tgt-sentence-length", type=int, default=1000)

    # Model parameters
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--multihead-attention-dropout", type=float, default=0.0)
    parser.add_argument("--normalize-before", type=bool_str, default=False)
    parser.add_argument("--activation", choices=['relu', 'gelu'], default='relu')
    parser.add_argument("--tgt-pos-enc-type", choices=['tree', 'seq', 'comb'], default=None)
    parser.add_argument("--max-path-depth", type=int, default=32)
    parser.add_argument("--path-multiple", type=int, default=0)
    parser.add_argument("--enable-copy", type=bool_str, default=True)
    parser.add_argument("--copy-att-layer", type=int, default=-1)
    parser.add_argument("--withcharemb", type=bool_str, default=False)
    parser.add_argument("--max-charseq-len", type=int, default=None)
    parser.add_argument("--share-nl-weights", type=bool_str, default=False)

    # Optimization parameters
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adam-beta-1", type=float, default=0.9)
    parser.add_argument("--adam-beta-2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-num-epochs", type=int, default=15)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-dataloader-workers", type=int, default=None)
    parser.add_argument("--distributed-backend",
                        choices=["dp", "ddp", "ddp_cpu", "ddp2", "ddp_spawn", "none"],
                        default="ddp")
    parser.add_argument("--log-gpu-memory", choices=["min_max", "all"], default=None)
    parser.add_argument("--detect-anomalies", action='store_true')

    # Scheduler
    parser.add_argument("--scheduler", choices=["steplr", "lambdalr"], default=None)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--sc-last-epoch", type=int, default=-1)

    # Trainer parameters
    parser.add_argument("--default-root-dir", type=str, default=None)

    # Testing parameters
    parser.add_argument("--target-language", choices=['python', 'astseq'], default='python')

    parser.add_argument("--loglevel", choices=['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'], default=None)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--random-seed", type=int_or_none, default=4831)

    return parser.parse_args()


def float_or_auto(val):
    return val if val in (None, "auto") else float(val)


def bool_str(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise ValueError('Unexpected bool value: ', val)


def int_or_none(val):
    return None if val == "none" else int(val)


def main():
    args = get_args()

    if args.random_seed is not None:
        seed_everything(args.random_seed)

    init_logging(args.loglevel, args.logfile, args.log)

    freeze_support()

    torch.set_printoptions(profile="full", linewidth=10000)
    log.info("Transformer args: %s", vars(args))

    if args.distributed_backend == 'none':
        args.distributed_backend = None

    if args.detect_anomalies:
        torch.autograd.set_detect_anomaly(True)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=True, save_last=True, save_top_k=5)

    trainer = Trainer(resume_from_checkpoint=args.resume_from_checkpoint,
                      max_epochs=args.max_num_epochs,
                      num_nodes=args.num_nodes,
                      gpus=args.gpus or args.num_gpus,
                      num_processes=args.num_processes,
                      distributed_backend=args.distributed_backend,
                      gradient_clip_val=args.gradient_clip_val,
                      weights_summary='full',
                      log_gpu_memory=args.log_gpu_memory,
                      track_grad_norm=2,
                      callbacks=[checkpoint_callback],
                      default_root_dir=args.default_root_dir,
                      progress_bar_refresh_rate=0)

    if args.num_dataloader_workers is not None:
        num_dataloader_workers = args.num_dataloader_workers
    # elif args.num_gpus:
    #    num_dataloader_workers = args.num_gpus
    else:
        num_dataloader_workers = 1

    start_time = datetime.now()
    log.info("Start Time = %s", start_time.strftime("%H:%M:%S"))

    if args.tgt_pos_enc_type is None:
        tgt_pos_enc_type = 'tree'
    else:
        tgt_pos_enc_type = args.tgt_pos_enc_type

    model_hparams = {
        'd_model': args.model_dim,
        'nhead': args.num_heads,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'multihead_attention_dropout': args.multihead_attention_dropout,
        'normalize_before': args.normalize_before,
        'activation': args.activation,
        'vocabsrc_size': args.vocabsrc_size,
        'vocabtgt_size': args.vocabtgt_size,
        'vocab_pad_id': 0,
        'max_path_depth': args.max_path_depth,
        'path_multiple': args.path_multiple,
        'tgt_pos_enc_type': tgt_pos_enc_type,
        'enable_copy': args.enable_copy,
        'copy_att_layer': args.copy_att_layer,
        'withcharemb': args.withcharemb,
        'vocabchar_size': args.vocabchar_size,
        'max_charseq_len': args.max_charseq_len,
        'share_nl_weights': args.share_nl_weights,
    }

    train_data_args = {
        'batch_size': args.batch_size,
        'num_dataloader_workers': num_dataloader_workers,
        'train_valid_data_path': args.train_valid_data_path,
        'train_data_path': args.train_data_path,
        'valid_data_path': args.valid_data_path,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'max_src_sentence_length': args.max_src_sentence_length,
        'max_tgt_sentence_length': args.max_tgt_sentence_length,
    }
    data_module = NL2CodeTrainDataModule(
        **train_data_args,
        max_path_depth=model_hparams['max_path_depth'],
        path_multiple=model_hparams['path_multiple'],
        max_charseq_len=model_hparams['max_charseq_len'] if model_hparams['withcharemb'] else None,
    )

    system = NL2CodeSystem(
        model_hparams=model_hparams,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        warmup_steps=args.warmup_steps,
        sc_last_epoch=args.sc_last_epoch,
        adam_beta_1=args.adam_beta_1,
        adam_beta_2=args.adam_beta_2,
        adam_epsilon=args.adam_epsilon,
        train_data_args=train_data_args,
    )

    trainer.fit(system, data_module)

    end_time = datetime.now()
    log.info("End Time = %s", end_time.strftime("%H:%M:%S"))
    log.info("Duration: %.1f sec.", (end_time - start_time).total_seconds())


def init_logging(loglevel, logfile_pattern, defaultlog):
    start_time = datetime.now()

    if defaultlog:
        logfile_pattern = 'transformer-%s.log'

    if loglevel is None:
        if logfile_pattern is not None:
            loglevel = 'DEBUG'
        else:
            loglevel = 'INFO'

    if logfile_pattern:
        logfile = logfile_pattern % start_time.strftime('%Y%m%d-%H%M%S')
        print("Logging to %s" % logfile)
        log.basicConfig(level=loglevel, filename=logfile, filemode='w')
        log.getLogger().addHandler(log.StreamHandler())
    else:
        log.basicConfig(level=loglevel)


if __name__ == '__main__':
    main()
