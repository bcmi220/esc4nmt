import os
import argparse
import logging
import sys
from tqdm import tqdm
from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data import data_utils
from fairseq.data.indexed_dataset import get_available_dataset_impl

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.debinarize')

def main(args):
    dictionary = Dictionary.load(args.dictionary)
    dataset = data_utils.load_indexed_dataset(args.dataset_path, dictionary, args.dataset_impl)

    logger.info('{} {} examples'.format(
        args.dataset_path, len(dataset)
    ))
    
    with open(args.output, 'w') as fout:
        for idx in tqdm(range(len(dataset))):
            tokens = dataset[idx]
            item_str = dictionary.string(tokens, args.remove_bpe)
            fout.write(item_str+'\n')


def cli_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    parser.add_argument("--dataset-path",
                       help="dataset path")
    parser.add_argument("--dictionary",
                       help="dictionary")
    parser.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                       help='remove BPE tokens before scoring (can be set to sentencepiece)')
    parser.add_argument('--output', type=str, metavar='FILE',
                       help='file to write to')
    args, _ = parser.parse_known_args()
    main(args)


if __name__ == '__main__':
    cli_main()