import argparse
import pickle

from copper_usage.model_factory import ModelFactory
from copper_usage.datamanager import TrainingsDataManager


def read_cmdl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_file', '-dcf', default='data_settings.yaml')
    parser.add_argument('--model_config_file', '-mcf', default='default_models.yaml')
    parser.add_argument('--input_file', '-i', required=True, help='file type must match configuration')
    parser.add_argument('--output_file_name', '-o', required=True)
    return vars(parser.parse_args())


def train(
    data_config_file: str,
    model_config_file: str,
    input_file: str,
    output_file_name: str,
    **kwargs
):
    model = ModelFactory.build_model_from_config(model_config_file)
    dmgr = TrainingsDataManager.init_from_config(
        model=model,
        file_path=input_file,
        cfg_file=data_config_file,
    )
    model.fit(dmgr.df)

    with open(output_file_name, 'wb') as pkl_out:
        pickle.dump(model, pkl_out)


def main():
    cmdl_args = read_cmdl_args()
    train(**cmdl_args)


if __name__ == '__main__':
    main()