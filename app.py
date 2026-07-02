# app.py
import streamlit as st
import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from pathlib import Path
from PIL import Image
from texts_config import TEXTS

from stg_colors import stg_color, ColorSet
from copper_usage.feature_containers import BoardFeatureContainer, MachineFeatureContainer
from copper_usage.model_factory import ModelFactory
from copper_usage.datamanager import TrainingsDataManager


PACKAGES_AVAILABLE = True

# Logo path - use relative path
logo_path = Path(__file__).resolve().parent / "Logo Blue Slogan.jpg"
logo_exists = logo_path.exists()

# Fixed safety margin - 5%
FIXED_MARGIN = 0.05

# Company colors with depth variations
COLOR_BLUE_1 = stg_color('blue', depth=1)    
COLOR_BLUE_2 = stg_color('blue', depth=2)
COLOR_BLUE_3 = stg_color('blue', depth=3)
COLOR_BLUE_4 = stg_color('blue', depth=4)

COLOR_GRAY_1 = stg_color('gray', depth=1)
COLOR_GRAY_2 = stg_color('gray', depth=2) 
COLOR_GRAY_3 = stg_color('gray', depth=3)  
COLOR_GRAY_4 = stg_color('gray', depth=4)   

COLOR_BLACK = stg_color('black', depth=1)    
COLOR_WHITE = stg_color('white', depth=1)    
COLOR_GREEN = stg_color('green', depth=1)   

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Wrapper function for bilingual texts
def get_text(key):
    """Get text in selected language"""
    return TEXTS.get(key, {}).get(st.session_state.language, f"Missing: {key}")


# Page configuration
st.set_page_config(
    page_title="JST Copper Plating Consumption Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Language selector in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Language / 语言")
    language = st.radio(
        "Select Language",
        options=['English', '中文'],
        index=0 if st.session_state.language == 'English' else 1,
        label_visibility="collapsed"
    )
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    st.markdown("---")

# CSS styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    
    .stApp {{ background: linear-gradient(135deg, {COLOR_GRAY_4}30 0%, {COLOR_WHITE} 100%); }}
    
    .main-header {{
        font-size: 2.5rem;
        color: {COLOR_BLUE_1};
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .result-card {{
        background: {COLOR_WHITE};
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid {COLOR_GRAY_4};
        transition: all 0.2s ease;
    }}
    
    .result-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-color: {COLOR_BLUE_3};
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLOR_BLUE_1};
        line-height: 1.2;
    }}
    
    .metric-label {{
        font-size: 0.8rem;
        color: {COLOR_GRAY_1};
        margin-top: 0.5rem;
        font-weight: 500;
    }}
    
    .metric-unit {{
        font-size: 0.7rem;
        color: {COLOR_GRAY_2};
        margin-top: 0.25rem;
    }}
    
    .stForm {{
        background: {COLOR_WHITE};
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid {COLOR_GRAY_4};
    }}
    
    .stButton > button {{
        background: {COLOR_BLUE_1};
        color: {COLOR_WHITE};
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background: {COLOR_BLUE_2};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,45,116,0.2);
    }}
    
    .stDownloadButton > button {{
        background: {COLOR_GRAY_1};
        color: {COLOR_WHITE} !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: {COLOR_BLUE_1};
    }}
    
    .footer {{
        text-align: center;
        color: {COLOR_GRAY_2};
        font-size: 0.75rem;
        padding: 1rem;
        margin-top: 1.5rem;
        border-top: 1px solid {COLOR_GRAY_4};
    }}
    
    .caption {{
        font-size: 0.7rem;
        color: {COLOR_GRAY_2};
        margin-top: -0.2rem;
    }}
    
    .note-badge {{
        background: {COLOR_BLUE_4}30;
        color: {COLOR_BLUE_1};
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_empty = st.columns([1, 5])
with col_logo:
    if logo_exists:
        img = Image.open(logo_path)
        max_width = 120
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        st.image(img, use_container_width=False)

st.markdown(f'<div class="main-header">{get_text("title")}</div>', unsafe_allow_html=True)

# File paths
TRAINING_DATA_PATH = Path(__file__).resolve().parent / "train_data.csv"
MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "src" / "copper_usage" / "config" / "default_models.yaml"
DATA_CONFIG_PATH = Path(__file__).resolve().parent / "src" / "copper_usage" / "config" / "data_settings.yaml"
MODEL_SAVE_PATH = Path(__file__).resolve().parent / "trained_model.pkl"

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_attempted' not in st.session_state:
    st.session_state.training_attempted = False

# Model loading/training - QUIETLY (no success messages)
if not st.session_state.model_loaded and PACKAGES_AVAILABLE and not st.session_state.training_attempted:
    st.session_state.training_attempted = True
    
    try:
        if MODEL_SAVE_PATH.exists():
            with open(MODEL_SAVE_PATH, 'rb') as f:
                st.session_state.model = pickle.load(f)
                st.session_state.model_loaded = True
        else:
            files_exist = all([
                TRAINING_DATA_PATH.exists(),
                MODEL_CONFIG_PATH.exists(),
                DATA_CONFIG_PATH.exists()
            ])
            
            if files_exist:
                with st.spinner(""):  # Empty spinner - will show nothing
                    model = ModelFactory.build_model_from_config(str(MODEL_CONFIG_PATH))
                    
                    # Load and prepare training data
                    dmgr = TrainingsDataManager.init_from_config(
                        model=model,
                        file_path=str(TRAINING_DATA_PATH),
                        cfg_file=str(DATA_CONFIG_PATH),
                    )
                    
                    model.fit(dmgr.df)
                    
                    with open(MODEL_SAVE_PATH, 'wb') as f:
                        pickle.dump(model, f)
                    
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                
    except Exception as e:
        # Only show error if model fails to load
        st.error(f"Model initialization error: {e}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        st.session_state.training_attempted = False

if not st.session_state.model_loaded:
    st.warning(f"### {get_text('model_not_available')}\n\n{get_text('ensure_files')}")
    st.stop()

# Input form
st.markdown(f"### {get_text('enter_specifications')}")

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(f"**{get_text('board_specifications')}**")
        
        is_vcp = st.selectbox(
            get_text('line_type'),
            options=[True, False],
            format_func=lambda x: get_text('vcp_line') if x else get_text('non_vcp_line')
        )
        
        board_thickness = st.number_input(
            f"{get_text('board_thickness')}",
            min_value=0.1,
            max_value=10.0,
            value=1.1,
            step=0.1,
            format="%.2f"
        )
        st.markdown(f'<div class="caption">{get_text("unit_mm")}</div>', unsafe_allow_html=True)
        
        ratio = st.number_input(
            get_text('aspect_ratio'),
            min_value=1.0,
            max_value=20.0,
            value=4.5,
            step=0.5,
            format="%.2f"
        )
        st.markdown(f'<div class="caption">{get_text("aspect_ratio_formula")}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"**{get_text('quality_requirements')}**")
        
        # Hole copper requirement with dropdown + custom option
        st.markdown(f"**{get_text('required_thickness')}**")
        
        # Options for dropdown
        requirement_options = ["15", "18", "20", "25", "Custom"]
        
        selected_requirement = st.selectbox(
            get_text('select_requirement'),
            options=requirement_options,
            key="requirement_select",
            help="Select standard requirement or choose 'Custom' to enter your own value"
        )
        
        # Initialize custom value in session state if not exists
        if 'custom_thickness' not in st.session_state:
            st.session_state.custom_thickness = 15.0
        
        if selected_requirement == "Custom":
            # Show number input for custom value
            minimal_thickness_um = st.number_input(
                f"{get_text('required_thickness')} ({get_text('unit_micrometer')})",
                min_value=5.0,
                max_value=100.0,
                value=st.session_state.custom_thickness,
                step=0.5,
                format="%.2f",
                key="custom_thickness_input",
                help="Enter customer required minimum copper thickness"
            )
            # Store the custom value in session state
            st.session_state.custom_thickness = minimal_thickness_um
        else:
            minimal_thickness_um = float(selected_requirement)
        
        st.markdown(f'<div class="caption">{get_text("unit_micrometer")}</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submitted = st.form_submit_button(get_text('calculate_button'), type="primary", use_container_width=True)
    
    if submitted:
        try:
            with st.spinner(get_text('calculating')):
                # Convert required thickness from μm to mm
                required_thickness_mm = minimal_thickness_um / 1000
                
                # Create BoardFeatureContainer with correct parameters
                board = BoardFeatureContainer(
                    is_vcp=is_vcp,
                    Ratio=ratio,
                    board_thickness=board_thickness,
                    required_thickness=required_thickness_mm,
                    margin=FIXED_MARGIN
                )
                
                # Make prediction
                result = st.session_state.model.predict(board)
                
                # Validate result
                if not isinstance(result, MachineFeatureContainer):
                    raise ValueError(f"Expected MachineFeatureContainer, got {type(result)}")
                
                # Store result in session state
                st.session_state.calculation_result = {
                    'result': result,
                    'is_vcp': is_vcp,
                    'board_thickness': board_thickness,
                    'ratio': ratio,
                    'minimal_thickness': minimal_thickness_um,
                    'margin': FIXED_MARGIN
                }
                
                st.rerun()
                
        except Exception as e:
            st.error(f"{get_text('calculation_error')}: {e}")
            with st.expander(get_text('technical_details')):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get('calculation_result'):
    data = st.session_state.calculation_result
    result = data['result']
    is_vcp = data['is_vcp']
    board_thickness = data['board_thickness']
    ratio = data['ratio']
    minimal_thickness_um = data['minimal_thickness']
    
    st.markdown("---")
    st.markdown(f"## {get_text('recommended_parameters')}")
    
    # Create 3 columns for metrics (removed target_thickness)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="metric-value">{result.plating_time:.1f}</div>
            <div class="metric-label">{get_text('plating_time')}</div>
            <div class="metric-unit">{get_text('minutes')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if hasattr(result, 'current_density_range') and result.current_density_range:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-value">{result.current_density_range[0]:.1f} - {result.current_density_range[1]:.1f}</div>
                <div class="metric-label">{get_text('current_density_range')}</div>
                <div class="metric-unit">{get_text('unit_a_cm2')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-value">{result.current_density:.2f}</div>
                <div class="metric-label">{get_text('current_density')}</div>
                <div class="metric-unit">{get_text('unit_a_cm2')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if is_vcp and hasattr(result, 'spray_frequency') and result.spray_frequency is not None:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-value">{result.spray_frequency:.1f}</div>
                <div class="metric-label">{get_text('spray_frequency')}</div>
                <div class="metric-unit">{get_text('unit_hz')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-value" style="font-size: 1.2rem;">{get_text('non_vcp_line')}</div>
                <div class="metric-label">{get_text('line_type')}</div>
                <div class="metric-unit">No spray control</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary
    st.markdown("---")
    st.markdown(f"### {get_text('process_summary')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📥 {get_text('input_parameters')}**
        
        - **{get_text('line_type')}:** {get_text('vcp_line') if is_vcp else get_text('non_vcp_line')}
        - **{get_text('board_thickness')}:** {board_thickness} mm
        - **{get_text('aspect_ratio')}:** {ratio:.2f}
        - **{get_text('required_thickness')}:** {minimal_thickness_um} μm
        - **Safety Margin:** {FIXED_MARGIN * 100}%
        """)
    
    with col2:
        output_text = f"**📤 {get_text('output_parameters')}**\n\n"
        output_text += f"- **{get_text('plating_time')}:** {result.plating_time:.1f} {get_text('minutes')}\n"
        
        if hasattr(result, 'current_density_range') and result.current_density_range:
            output_text += f"- **{get_text('current_density_range')}:** {result.current_density_range[0]:.1f} - {result.current_density_range[1]:.1f} {get_text('unit_a_cm2')}\n"
        else:
            output_text += f"- **{get_text('current_density')}:** {result.current_density:.2f} {get_text('unit_a_cm2')}\n"
        
        if is_vcp and hasattr(result, 'spray_frequency') and result.spray_frequency is not None:
            output_text += f"- **{get_text('spray_frequency')}:** {result.spray_frequency:.1f} {get_text('unit_hz')}"
        
        st.markdown(output_text)
    
    # Export
    export_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_vcp": is_vcp,
        "Ratio": ratio,
        "board_thickness_mm": board_thickness,
        "required_thickness_um": minimal_thickness_um,
        "safety_margin_percent": FIXED_MARGIN * 100,
        "plating_time_min": result.plating_time,
    }
    
    if hasattr(result, 'current_density_range') and result.current_density_range:
        export_data["current_density_range_min_A_cm2"] = result.current_density_range[0]
        export_data["current_density_range_max_A_cm2"] = result.current_density_range[1]
    else:
        export_data["current_density_A_cm2"] = result.current_density
    
    # Keep target_thickness in export but remove from display
    if hasattr(result, 'target_thickness') and result.target_thickness:
        export_data["target_thickness_um"] = result.target_thickness
    
    if is_vcp and hasattr(result, 'spray_frequency') and result.spray_frequency is not None:
        export_data["spray_frequency_Hz"] = result.spray_frequency
    
    export_df = pd.DataFrame([export_data])
    csv = export_df.to_csv(index=False)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label=get_text('download_results'),
            data=csv,
            file_name=f"machine_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown(f"""
<div class="footer">
    <div>{get_text('footer')}</div>
    <div style="font-size: 0.65rem;">{get_text('powered_by')}</div>
</div>
""", unsafe_allow_html=True)