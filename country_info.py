import pandas as pd

from pathlib import Path

from dataclasses import dataclass


@dataclass
class CountryInfoColumns:
    name: str="ISO_name"
    sovereignty: str="Sovereignty"
    two_letter_code: str="two_letter_code"
    three_letter_code: str="three_letter_code"
    number_code: str="number_code"
    subdiv_code: str="ISO 3166-2 subdivision codes link"
    web_ending: str="TLD"
    continent: str="Continent_Name"


class CountryInfo:

    def __init__(
            self, 
            df: pd.DataFrame,
            config: CountryInfoColumns,
        ):
        self.df = df
        self.config = config

    @classmethod
    def init_from_csv(
            cls, 
            file_path: Path=None, 
            config_dict: dict=None,
            sep=';',
            **load_kwargs,
        ):
        default_file_path = Path(__file__).parent / 'csv_data'\
              / 'iso_country_data.csv'
        config = CountryInfoColumns(**(config_dict if config_dict is not None else {}))
        df = pd.read_csv(
            default_file_path if file_path is None else file_path, 
            sep=sep, 
            **load_kwargs
        )
        return cls(df, config)
    
    def get(self, column_code: str, index: str='two_letter_code') -> pd.Series:

        assert hasattr(self.config, column_code)
        assert hasattr(self.config, index)

        return self.df.set_index(getattr(self.config, index))[getattr(self.config, column_code)]