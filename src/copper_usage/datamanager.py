from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from copper_usage.model import Model

from pathlib import Path

import yaml


class PretransformationStep(ABC):

    column: str=None

    @abstractmethod
    def apply(self, df: pd.DataFrame, *args, **kwargs):
        pass


PRETRANSFORMATION_REGISTRY = {}


def register_pretransformation_step(name: str=None):
    def decorator(cls):
        key = name or cls.__name__.lower()
        PRETRANSFORMATION_REGISTRY[key] = cls
        return cls
    return decorator


@register_pretransformation_step('enforce_float')
class EnforceFloat(PretransformationStep):

    def __init__(self, columns: str | list[str], *args, **kwargs):
        if isinstance(columns, str):
            self.columns = [columns]
        elif isinstance(columns, list):
            self.columns = columns
        else:
            raise TypeError

    def _enforce(self, X):
        try:
            return float(X)
        except ValueError:
            return np.nan

    def apply(self, df, *args, **kwargs):

        columns = list(set(df.columns) & set(self.columns))

        for column in columns:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                df[column] = df[column].apply(self._enforce)
        return df
        
    

class Pretransformations:

    def __init__(self, steps: list[PretransformationStep]):
        self.steps = steps

    @classmethod
    def init_from_config(cls, cfg, raise_if_missing: bool=True):

        steps = []

        for key, values in cfg.items():
            try:
                steps.append(
                    PRETRANSFORMATION_REGISTRY[key](**values) 
                    if isinstance(values, dict)
                    else PRETRANSFORMATION_REGISTRY[key](values)
                )
            except KeyError as e:
                if raise_if_missing:
                    raise KeyError(e)
                else:
                    continue
        
        return cls(steps)

    def apply(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        for step in self.steps:
            df = step.apply(df, *args, **kwargs)
        return df


class TrainingsDataManager:
    
    def __init__(self, df):
        self._df = df

    @staticmethod
    def replace_value():
        pass

    @classmethod
    def init_from_file_and_model(
        cls, 
        model: Model,
        file_path: str,
        cfg: dict,
        file_type: str='csv',
        load_kwargs: dict=None,
        *args,
        **kwargs,
    ):
        if file_type == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path, **(load_kwargs or {}))
        else:
            raise TypeError(f'{file_path} is not a valid file_ending')
        
        for column in model.get_mandatory_not_null():
            df = df[~df[column].isnull()]

        pretrs = Pretransformations.init_from_config(cfg)
        df = pretrs.apply(df)

        return cls(df=df)
    
    @property
    def df(self):
        return self._df
    
    @classmethod
    def init_from_config(
        cls, 
        model: Model,
        file_path,
        cfg_file,
        **kwargs,
    ):

        cfg_path = Path(cfg_file)
        cfg_path = Path(__file__).resolve().parent / Path('config') / cfg_file \
            if not cfg_path.is_file() else cfg_path

        with open(cfg_path, 'r') as yin:

            return cls.init_from_file_and_model(
                model=model,
                file_path=file_path,
                **yaml.safe_load(yin),
            )