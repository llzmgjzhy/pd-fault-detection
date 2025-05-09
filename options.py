import argparse


class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description="Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments."
        )

        # basic config
        self.parser.add_argument(
            "--task",
            choices={
                "fault_detection",
                "classification",
                "forecast",
            },
            default="fault_detection",
            help=(
                "Training objective/task: fault detection of entire time series\n"
                "classification of entire time series,\n"
                "forecasting of future values"
            ),
        )
        self.parser.add_argument(
            "--output_dir",
            default="./experiments",
            help="Root output directory. Must exist. Time-stamped directories will be created inside.",
        )
        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--records_file",
            default="./records.xlsx",
            help="Excel file keeping all records of experiments",
        )
        self.parser.add_argument(
            "--comment",
            type=str,
            default="",
            help="A comment/description of the experiment",
        )

        # data loader
        self.parser.add_argument(
            "--root_path",
            type=str,
            default="./dataset",
            help="root path of the data file",
        )
        self.parser.add_argument(
            "--meta_path",
            type=str,
            default="vsb-power-line-fault-detection",
            help="data file",
        )
        self.parser.add_argument(
            "--data_path",
            type=str,
            default="three-phase-denoise-features",
            help="data file",
        )

        # forecasting task
        self.parser.add_argument(
            "--seq_len", type=int, default=800000, help="input sequence length"
        )
        self.parser.add_argument(
            "--label_len", type=int, default=48, help="start token length"
        )
        self.parser.add_argument(
            "--pred_len", type=int, default=512, help="prediction sequence length"
        )
        self.parser.add_argument(
            "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
        )

        # System
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=1,
            help="Print batch info every this many batches",
        )
        self.parser.add_argument(
            "--gpu", type=str, default="0", help="GPU index, -1 for CPU"
        )
        self.parser.add_argument(
            "--n_proc",
            type=int,
            default=-1,
            help="Number of processes for data loading/preprocessing. By default, equals num. of available cores.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=2025,
            help="Seed used for splitting sets. None by default, set to an integer for reproducibility",
        )
        self.parser.add_argument(
            "--no_timestamp",
            action="store_true",
            help="If set, a timestamp will not be appended to the output directory name",
        )

        # Training process
        self.parser.add_argument(
            "--epochs", type=int, default=100, help="Number of training epochs"
        )
        self.parser.add_argument("--itr", type=int, default=3)
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=2,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )
        self.parser.add_argument(
            "--optimizer",
            choices={"Adam", "AdamW", "RAdam"},
            default="Adam",
            help="Optimizer",
        )
        self.parser.add_argument(
            "--patience", type=int, default=3, help="early stopping patience"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate (default holds for batch size 64)",
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="weight decay (L2 penalty) (default: 0.0)",
        )
        self.parser.add_argument(
            "--lr_step",
            type=str,
            default="1000000",
            help="Comma separated string of epochs when to reduce learning rate by a factor of 10."
            " The default is a large value, meaning that the learning rate will not change.",
        )
        self.parser.add_argument(
            "--lr_factor",
            type=str,
            default="0.1",
            help=(
                "Comma separated string of multiplicative factors to be applied to lr "
                "at corresponding steps specified in `lr_step`. If a single value is provided, "
                "it will be replicated to match the number of steps in `lr_step`."
            ),
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=64, help="Training batch size"
        )
        self.parser.add_argument(
            "--key_metric",
            choices={"loss", "accuracy", "precision", "mcc", "mse"},
            default="loss",
            help="Metric used for defining best epoch",
        )
        self.parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )

        # Model
        self.parser.add_argument(
            "--patch_size", type=int, default=1000, help="patch_size"
        )
        self.parser.add_argument("--stride", type=int, default=1000, help="stride")
        self.parser.add_argument(
            "--model_name",
            default="patchtst",
            help="Model class",
        )
        self.parser.add_argument(
            "--d_model",
            type=int,
            default=128,
            help="Internal dimension of transformer embeddings",
        )
        self.parser.add_argument(
            "--n_layers",
            type=int,
            default=3,
            help="Number of transformer encoder layers (blocks)",
        )
        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout applied to most transformer encoder layers",
        )
        self.parser.add_argument(
            "--pos_encoding",
            choices={"fixed", "learnable"},
            default="fixed",
            help="Internal dimension of transformer embeddings",
        )
        self.parser.add_argument(
            "--activation",
            choices={"relu", "gelu"},
            default="gelu",
            help="Activation to be used in transformer encoder",
        )
        self.parser.add_argument(
            "--enc_in",
            type=int,
            default=3,
            help="Number of input features (channels) in the input time series",
        )
        self.parser.add_argument(
            "--split_num",
            type=int,
            default=5,
            help="Dropout applied to most transformer encoder layers",
        )
        self.parser.add_argument(
            "--loss",
            choices={"cross_entropy", "focal", "mse"},
            default="cross_entropy",
            help="loss used for train model",
        )
        self.parser.add_argument(
            "--d_ff",
            type=int,
            default=512,
            help="dimension of fcn",
        )
        self.parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        self.parser.add_argument(
            "--llm_dim", type=int, default="768", help="LLM model dimension"
        )  # LLama7b:4096; GPT2-small:768; BERT-base:768; Qwen_2_5_VL:3584
        self.parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        self.parser.add_argument("--percent", type=int, default=100)

        # image modality
        self.parser.add_argument(
            "--img_width", type=int, default=256, help="the width of time series image"
        )
        self.parser.add_argument(
            "--img_height",
            type=int,
            default=256,
            help="the height of time series image",
        )

    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(",")]
        args.lr_factor = [float(i) for i in args.lr_factor.split(",")]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor
        ), "You must specify as many values in `lr_step` as in `lr_factors`"

        return args
