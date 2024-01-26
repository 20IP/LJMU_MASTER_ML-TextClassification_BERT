import argparse
import warnings 
from modeling import MedicalTextClassifier
from dataset import MedicalTextDataLoader
from logger_config import logger
warnings.filterwarnings('ignore') 

def main():

    parser = argparse.ArgumentParser(description="Initlizing coding")
    parser.add_argument("--data_path",
                        type=str,
                        help="data path include train+test file",
                        default='/home/dev/Xavier/LJMU/Medical-Abstracts-TC-Corpus')
    parser.add_argument("--data_preprocess",
                        type=bool,
                        help="requirement do or dont pre-processing data",
                        default=False)
    parser.add_argument("--model_pretrain",
                        type=str,
                        help="chose the model to fine-tuning",
                        default='albert_base_v2',
                        required=False
                        )
    parser.add_argument("--model_dir",
                        type=str,
                        help="Directory load local pre-train",
                        default='/home/dev/Xavier/LJMU/pre-train',
                        required=False
                        )
    parser.add_argument("--loss_type",
                        type=str,
                        help="chose the loss function to fine-tuning",
                        default='',
                        required=False
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=8
                        )
    parser.add_argument("--epochs",
                        type=int,
                        default=5
                        )
    parser.add_argument("--step_per_epoch",
                        type=int,
                        default=100
                        )
    parser.add_argument("--max_length",
                        type=int,
                        default=512
                        )
    parser.add_argument("--reduce_learing_rate",
                        type=bool,
                        default=False
                        )
    parser.add_argument("--scheduler",
                        type=bool,
                        default=False
                        )
    parser.add_argument("--truncate",
                        type=bool,
                        default=True
                        )
    parser.add_argument("--padding",
                        type=bool,
                        default=True
                        )
    parser.add_argument("--report_method",
                        type=str,
                        default='micro'
                        )
    parser.add_argument("--reduce_step_size",
                        type=int,
                        default=2
                        )
    parser.add_argument("--reduce_gamma",
                        type=float,
                        default=0.5
                        )
    args = parser.parse_args()
    logger.info(f"\n ************ Model Name: {args.model_pretrain} - Loss type: {args.loss_type} **************\n")
    
    data_loader = MedicalTextDataLoader(args)
    data_train, data_test, num_labels = data_loader.load_data()
    
    classifier = MedicalTextClassifier(args, num_labels)
    classifier.fit_data(data_train, data_test)
    classifier.train_and_evaluate()


if __name__ == "__main__":
    main()
