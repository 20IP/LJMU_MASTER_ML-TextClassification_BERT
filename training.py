import argparse
from modeling import MedicalTextBase, MedicalTextDataLoader


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
                        default='bert-base-uncased',
                        required=False
                        )
    parser.add_argument("--loss_function",
                        type=str,
                        help="chose the loss function to fine-tuning",
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
    parser.add_argument("--average_report",
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
    

    data_loader = MedicalTextDataLoader(args)
    data_train, data_test = data_loader.load_data()

    
    classifier = MedicalTextBase(model_name=args.model_pretrain,
                                 lr = args.learning_rate,
                                 batch_size=args.batch_size,
                                 step_per_epoch=args.step_per_epoch
                                 )
    classifier.load_model()
    classifier.fit_data(data_train, data_test)
    classifier.initialize_optimizer_scheduler()
    classifier.train_and_evaluate()

    
    
if __name__ == "__main__":
    main()
