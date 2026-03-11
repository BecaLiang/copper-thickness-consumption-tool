import pandas as pd


from pydantic.dataclasses import dataclass


from copper_usage.model import ThicknessCalculation


class DataManager():
    pass


class ModelManager:
    
    def __init__(self, calculators: list[ThicknessCalculation]):
        self.calculators = calculators

    def fit(self, ):
        pass

    def calculate_parameters(self):
        pass