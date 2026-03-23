import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from copper_usage.model import Model
from copper_usage.thickness_calculation import (
    DataColumns,
    PlainLinearThicknessCalculation,
    BoardInclusiveLinearThicknessCalculation,
    PlainLinearModel,
    PlainLinearModelWithBoard
)
from copper_usage.sop_slicer import SlicerFactory
from copper_usage.inverter import GaussianErrorModel


st.set_page_config(
    page_title="Copper Thickness Recommender",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Copper Plating Thickness Prediction Model (Testing)")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")