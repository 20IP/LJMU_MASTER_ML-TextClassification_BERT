import argparse


parser = argparse.ArgumentParser(
                    prog='Bert for Medical Text classify',
                    description='Improve based o n Fine-tuning Model',
                    epilog='Text at the bottom of help')

parser.add_argument('--epochs', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--smoothing', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--pretrain-dir', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--batch-size', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--learing-rate', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--datdaset-train', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--datdaset-test', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--output', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--re-processing', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--average-type', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--max_length', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--step-size', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--gamma', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument('--weight_decay', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')


def main():
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='JSON file with config parameters')
    parser.add_argument('model_name', help='name under which model will be stored')
    parser.add_argument('-dual', default=False, action='store_true', help='True if dual scoring is used, False otherwise')
    parser.add_argument('-global_info', default=False, action='store_true', help='True if global information is used for context representation, False otherwise')
    parser.add_argument('-weighted_sum', default=False, action='store_true', help='True if sum of scores is weighted, False otherwise')
    args = parser.parse_args()

    main(args)
