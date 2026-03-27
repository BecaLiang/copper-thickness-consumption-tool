# Copper Usage

This projects provides a tool to calculate machine parameters for the pattern plating step at JST. Customer provided inputs like the desired thickness are combined with a safety margin to answer the question "What are my machine settings so I fulfill the customer required minimal thickness with  a probability of 1 - margin?". To achieve this, a layer modelling the effect of our process on the minimal copper thickness is also contained.

## Installation

simply use pip in a virtual env of your choice. Clone this repository and then with an active virtual env and on windows, run
`python -m pip install -e .`

## Training

### CLI-Example

_Example with default configs_:
`python .\bin\train_example.py -i .\train_data.csv -o testout.pkl`
Here, train_data.csv need to come from the factory. testout.pkl is a serialized instance of the _Model_-class which can be used to predict machine parameters. See the step __Application__.

## Application

### Interface

The _predict_ method of the _Model_-class expects an instance of _BoardFeatureContainer_ which contains all necessary board informations. Those are as of writing this Readme:
  * is_vcp: (whether the panel plating has been done on a vcp line)
  * Ratio: (Ratio between thickness and whole diameter)
  * board_thickness: thickness of the board in mm
  * minimal_thickness: customer required minimal thickness
  * margin: max. probability for the process to result in a thickness smaller than the _minimal_thickness_   
 The result of the algorithm will be output in the form of an instance of _MachineFeatureContainer_. This class holds the following attributes:
  * plating_time: duration in minutes
  * current_density: current density in ??? (A/cm^2?)
  * spray_frequency: in case of _is\_vcp_, the spray frequency in Hz. Can be _None_.

### CLI-Example

_Example with default configs_:
`python .\bin\predict_example.py --margin 0.05 --minimal_thickness 15 --is_vcp True --Ratio 4.5 --board_thickness 1.1 --model_file .\testout.pkl`

