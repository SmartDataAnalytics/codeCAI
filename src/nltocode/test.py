import argparse
import logging as log
from datetime import datetime
from multiprocessing import freeze_support

import torch
from pytorch_lightning import Trainer, seed_everything

from nltocode.datamodule import NL2CodeTestDataModule
from nltocode.nl2code import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser("Test model", fromfile_prefix_chars='@')

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-dataloader-workers", type=int, default=1)
    parser.add_argument("--distributed-backend",
                        choices=["dp", "ddp", "ddp_cpu", "ddp2", "ddp_spawn", "none"],
                        default="ddp")
    parser.add_argument("--log-gpu-memory", choices=["min_max", "all"], default=None)
    parser.add_argument("--detect-anomalies", action='store_true')

    # Testing parameters
    parser.add_argument("--default-root-dir", type=str, default=None)
    parser.add_argument("--test-model-path", type=str, default=None)
    parser.add_argument("--test-data-path", type=str, default="test-data.json")
    parser.add_argument("--grammar-graph-file", type=str, default='pythongrammar.json')
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--disable-decoder-constraint-mask", action='store_true')
    parser.add_argument("--max-beam-length", type=int, default=None)
    parser.add_argument("--max-num-predicted-results", type=int, default=1)
    parser.add_argument("--beam-search-mode", choices=['full', 'reduced', 'scaled'], default='full')
    parser.add_argument("--treat-empty-beamsearch-results-as-invalid", type=bool_str)
    parser.add_argument("--keep-invalid-beamsearch-results", action='store_true')
    parser.add_argument("--validate-parsability", type=bool_str, default=True)
    parser.add_argument("--report-beams-every-n-tokens", type=int, default=None)
    parser.add_argument("--report-k-best-beams", type=int, default=None)
    parser.add_argument("--target-output-file", type=str)
    parser.add_argument("--target-language", choices=['python', 'astseq'], default='python')

    parser.add_argument("--loglevel", choices=['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'], default=None)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--random-seed", type=int_or_none, default=4831)

    return parser.parse_args()


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

    trainer = Trainer(num_nodes=args.num_nodes,
                      gpus=args.gpus or args.num_gpus,
                      num_processes=args.num_processes,
                      distributed_backend=args.distributed_backend,
                      weights_summary='full',
                      log_gpu_memory=args.log_gpu_memory,
                      default_root_dir=args.default_root_dir,
                      progress_bar_refresh_rate=0)

    start_time = datetime.now()
    log.info("Start Time = %s", start_time.strftime("%H:%M:%S"))

    test_args = {
        'grammar_graph_file': args.grammar_graph_file,
        'target_language': args.target_language,
        'num_beams': args.num_beams,
        'disable_decoder_constraint_mask': args.disable_decoder_constraint_mask,
        'max_beam_length': args.max_beam_length,
        'max_num_predicted_results': args.max_num_predicted_results,
        'beam_search_mode': args.beam_search_mode,
        'treat_empty_beamsearch_results_as_invalid': args.treat_empty_beamsearch_results_as_invalid,
        'keep_invalid_beamsearch_results': args.keep_invalid_beamsearch_results,
        'validate_parsability': args.validate_parsability,
        'report_beams_every_n_tokens': args.report_beams_every_n_tokens,
        'report_k_best_beams': args.report_k_best_beams,
        'target_output_file': args.target_output_file,
    }
    system = load_checkpoint(args.test_model_path, test_args)
    trainer.model = system

    data_module = NL2CodeTestDataModule(
        batch_size=args.batch_size,
        num_dataloader_workers=args.num_dataloader_workers,
        test_data_path=args.test_data_path,
        is_test_only_run=True,
        max_charseq_len=system.model.max_charseq_len if system.model.withcharemb else None
    )

    trainer.test(ckpt_path=None, datamodule=data_module)

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
