import pickle
import simple_parsing

from copper_usage.feature_containers import (
    BoardFeatureContainer,
    MachineFeatureContainer,
)
from copper_usage.model import Model


def read_cmdl_args():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(BoardFeatureContainer, dest='bfc')
    parser.add_argument('--model_file', required=True)
    return parser.parse_args()


def main():
    cmdl_args = read_cmdl_args()
    with open(cmdl_args.model_file, 'rb') as pkl_in:
        model = pickle.load(pkl_in)
    print(model.predict(cmdl_args.bfc))


if __name__ == '__main__':
    main()