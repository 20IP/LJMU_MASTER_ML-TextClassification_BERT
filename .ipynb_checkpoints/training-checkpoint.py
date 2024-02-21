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
                        default='../Medical-Abstracts-TC-Corpus')
    
    parser.add_argument("--data_preprocess",
                        type=bool,
                        help="requirement do or dont pre-processing data",
                        default=False,
                        required=True)
    
    parser.add_argument("--data_lemma",
                        type=bool,
                        default=False,
                        required=True)
    
    parser.add_argument("--model_pretrain",
                        type=str,
                        default='clinicalBERT',
                        choices=['bert-base-uncased','roberta-base',
                                 'bluebert_pubmed_uncased','bioBERT_v1.1',
                                 'clinicalBERT','alBERT_base_v2'],
                        help="chose the model to fine-tuning",
                        required=True
                        )
    
    parser.add_argument("--model_dir",
                        type=str,
                        help="Directory load local pre-train",
                        default='../pre-train',
                        required=False
                        )
    
    parser.add_argument("--output_dir",
                        type=str,
                        help="Directory for save fine-tuning model",
                        default='../output-fine-tuning',
                        required=False
                        )
    
    parser.add_argument("--loss_type",
                        type=str,
                        help="chose the loss function to fine-tuning",
                        default='ce',
                        choices=['ce', 'fcl', 'fclbnl2', 'lbsmoothingloss'],
                        required=True
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
                        default=200
                        )
    parser.add_argument("--max_length",
                        type=int,
                        default=512
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
                        default='micro',
                        choices=['micro']
                        )
    
    parser.add_argument("--reduce_step_size",
                        type=int,
                        default=2
                        )
    
    parser.add_argument("--reduce_gamma",
                        type=float,
                        default=0.2
                        )
    
    args = parser.parse_args()
    
    logger.info('\t ########################################################## \n')
    logger.info('\t'*3 + 'INFOMATION OF CONFIGURATION:')
    logger.info(args)
    
    data_loader = MedicalTextDataLoader(args)
    
    logger.info('\t Loading and Processing training data. \n')
    data_train, num_labels = data_loader.load_data(data_type='train')
    logger.info('\t Loading and Processing testing data. \n')
    data_test, _ = data_loader.load_data(data_type='test')
    
    classifier = MedicalTextClassifier(args, num_labels)
    classifier.fit_data(data_train, data_test)
    classifier.train_and_evaluate()


if __name__ == "__main__":
    main()
