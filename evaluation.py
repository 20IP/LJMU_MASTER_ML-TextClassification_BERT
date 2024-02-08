import argparse
import warnings 
from modeling import MedicalTextClassifier
from dataset import MedicalTextDataEvalLoader
from logger_config import configure_logger
warnings.filterwarnings('ignore')

logger_eval = configure_logger('Evaluation logger.log')

parser = argparse.ArgumentParser(description="Initlizing coding")
parser.add_argument("--data_path",
                        type=str,
                        help="data path include train+test file",
                        default='../Medical-Abstracts-TC-Corpus')

parser.add_argument("--model_dir",
                    type=str,
                    help="Directory load local output-fine-tuning",
                    default='../output-fine-tuning',
                    required=False
                    )

parser.add_argument("--model_fine_tuning",
                    type=str,
                    default='roberta-base-True-lbsmoothingloss-False.pth',
                    help="chose the model to fine-tuning",
                    required=False
                    )

parser.add_argument("--report_method",
                        type=str,
                        default='micro',
                        choices=['micro']
                        )

args = parser.parse_args()

def evaluate():
    model_name = args.model_fine_tuning
    data_path = args.data_path
    data_preprocess, loss_type, _ = model_name.replace('.pth', '').split('-')[::-1][-3:]
    
    if data_preprocess == 'True':
        data_preprocess = True
    else:
        data_preprocess = False
        
    data_loader = MedicalTextDataEvalLoader(data_path, data_preprocess)
    test_data = data_loader.load_data()

evaluate()