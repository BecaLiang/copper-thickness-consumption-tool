# Copper Usage

This projects provides a tool to calculate machine parameters for the pattern plating step at JST. Customer provided inputs like the desired thickness are combined with a safety margin to answer the question "What are my machine settings so I fulfill the customer required minimal thickness with a probability of 1 - margin?". To achieve this, a layer modelling the effect of our process on the minimal copper thickness is also contained.

## 1. Working Principles

The copper thickness tool is based on two functional entities. Firstly a tool to calculate the minimum thickness among all 6 measured points from all relevant inputs, implemented as an inheritance from _ThicknessCalculation_ in _thickness\_calculation.py_. Secondly, an optimization mechanism that uses that exact function to calculate the theoretical minimal thickness _(tmt)_ and selects the parameterset with the lowest theroretical tmt fulfilling a pre-set safety margin. This is achieved by essentially inverting the _ThicknessCalculation_, i.e. taking a given thickness and backstrapolating a set of parameters which yield said value. Consequently, the task is taken by a daughter of _ErrorModel_ from _inverter.py_.

### 1.2 ThicknessCalculation

To calculate the most likely tmt, we need a function mapping the given set of input variables to the output, i.e. the tmt. <br>

$$
\begin{align*}
  \text{tmt}:\hspace{1.5em} t^{\text{Cu}}_{\text{min}} = F\left(j, T_{\text{plate}}, f_{\text{spray}}, t_{\text{board}}\right) \\ 
\end{align*}
$$

<div style="margin-top:15px;"></div>

The relevant implementation of _F_ depends on the following parameters: <br>

$$
\begin{align*}
  j &: \text{Current Density in } \frac{\text{A}}{\text{cm}^2} & T_{\text{plate}} &: \text{Plating Time in min}\\
  f_{\text{spray}} &: \text{Spray Frequency in Hz} & t_{\text{board}} &: \text{board thickness in mm}\\
\end{align*}
$$

#### 1.2.1 Implementation Details

Different functions are to be implemented as daughter classes of _ThicknessCalculation_. The mother class provides _fit_ and _predict_ as interface functions. calling either with a pandas DataFrame as argument works analogous to the very common scikitlearn-type interfacing. _fit_ takes a set of variable values with correclty assigned tmt-values, taken from the real machines and then internally calculates the best fitting parameters. _predict_ then uses those to calculate the tmt from any given input. On a technical level, each new instance needs to provide _\_fit_ and _\_predict_ methods to do this. _fit_ and _predict_ are the outside interface which also implement such things as cleaning the data.

The selection of each model is implemented following the registry pattern. The concrete class has to be decorated with _register\_calculator_ like so. 

```
@register_calculator('my_implementation')
class MyImplementation(ThicknessCalculatin):

    def _fit
        ...

    def _predict
        ...
```

_my\_implementation_ can now be used in configurations / calls e.g. to this modules factory-class _ModeFactory_ from _model\_factory.py_.

#### 1.2.2 The linear model

To achieve a stable and comprehensible performance, the default version uses a plain linear model. No higher order dependencies than first order are used. Thorough investigation into the dependencies of the model on the parameters showed a consistent linear behaviour of tmt as a function of both time and current density. The implementation for the vcp line is done in _BoardInclusiveLinearThicknessCalculation_.

### 1.2 Slicing

The original SOP (**S**tandard **O**perating **P**rocedure) gives values for the machine parameters dependent on the Ratio between the board thickness in mm and the smallest drill diameter, also in mm, further distinguished by different versions for vcp and non-vcp lines. To keep track and keep the structure of the state of the art method, different internal parameters are trained for distinguished slices along the _Ratio_ and _is\_vcp_ axes. Slicing is implemented as an an instance of _SOPSlicer_, constructed with upper and lower bound, where the lower border is part of the set and the upper is not. Configuration happens under the _slices_ keyword as a simple list of boarders for the vcp and non-vcp line, respectively. 

### 1.3 Inversion

As outlined above, to work out the optimal values for machine parameters, the inverted operation of th fit to the _ThicknessCalculation_ instance needs to be accomplished. To achieve that, a second part is needed and that is in addition to a model for the expected value, a model for the variance, or in other terms the center and width of the target distribution. Examinations have shown the individual tmts to be normally distributed. Thus, a gaussian dependency can savely be assumed, implemented via _GaussianErrorModel_ in _inverter.py_. 

By default, the safety margin $M_{\text{safe}}$ is chosen to be 5% which of course is additive to the result of the _ThicknessCalculation_ at the operating point. The customer provided required thickness $t_{\text{req}}$ plus the margin equals the recommended thickness $t_{\text{oc}}$

$$
t_{\text{req}} + M = t_{\text{oc}}
$$

The concrete values are simply extracted from the cumulative density function of the normal distribution _G_, A call to an optimzer with a simple quadratic metric $m_{\text{min}}$ ensures the correctness of this choise.

$$
 \left(G - M\right)^2 = m_{\text{min}}
$$

## 2. Installation

simply use pip in a virtual env of your choice. Clone this repository and then with an active virtual env and on windows, run

`python -m pip install -e .`

## Training

### CLI-Example

_Example with default configs_:<br>
```python .\bin\train_example.py -i .\train_data.csv -o testout.pkl```
Here, train_data.csv need to come from the factory. testout.pkl is a serialized instance of the _Model_-class which can be used to predict machine parameters. See the step __Application__.

## 3. Application

### 3.1 Interface

The _predict_ method of the _Model_-class expects an instance of _BoardFeatureContainer_ which contains all necessary board informations. Those are as of writing this Readme:
  * is_vcp: (whether the panel plating has been done on a vcp line)
  * Ratio: (Ratio between thickness and whole diameter)
  * board_thickness: thickness of the board in mm
  * minimal_thickness: customer required minimal thickness
  * margin: max. probability for the process to result in a thickness smaller than the _minimal_thickness_   
 The result of the algorithm will be output in the form of an instance of _MachineFeatureContainer_. This class holds the following attributes:
  * plating_time: duration in minutes
  * current_density: current density in A/cm^2?
  * target_thickness: the most likely minimal thickness in µm given all other parameters.
  * spray_frequency: in case of _is\_vcp_, the spray frequency in Hz. Can be _None_.

In addition, _MachineFeatureContainer_ exhibits as well the property _current\_density\_range_. This property gives a range, 1 A/cm^2 wide, rounded to .5 A/cm^2, around the current density value identified as optimal. It is formatted as tuple of two floats.

### 3.2 CLI-Example

_Example with default configs_:<br>
```python .\bin\predict_example.py --margin 0.05 --minimal_thickness 15 --is_vcp True --Ratio 4.5 --board_thickness 1.1 --model_file .\testout.pkl```


# The Streamlit App