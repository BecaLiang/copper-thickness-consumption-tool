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

# Add paths - use relative paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

from stg_colors import stg_color, ColorSet
from copper_usage.feature_containers import BoardFeatureContainer, MachineFeatureContainer
from copper_usage.model_factory import ModelFactory
from copper_usage.datamanager import TrainingsDataManager

PACKAGES_AVAILABLE = True

# Logo path - use relative path
logo_path = current_dir / "Logo Blue Slogan.jpg"
logo_exists = logo_path.exists()

# Fixed safety margin - 5%
FIXED_MARGIN = 0.05

# Company colors with depth variations
COLOR_BLUE_1 = stg_color('blue', depth=1)      # '#002D74'
COLOR_BLUE_2 = stg_color('blue', depth=2)      # '#0051D1'
COLOR_BLUE_3 = stg_color('blue', depth=3)      # '#3F89FF'
COLOR_BLUE_4 = stg_color('blue', depth=4)      # '#A1C5FF'

COLOR_GRAY_1 = stg_color('gray', depth=1)      # '#606163'
COLOR_GRAY_2 = stg_color('gray', depth=2)      # '#898A8D'
COLOR_GRAY_3 = stg_color('gray', depth=3)      # '#B2B3B7'
COLOR_GRAY_4 = stg_color('gray', depth=4)      # '#D4D6DB'

COLOR_BLACK = stg_color('black', depth=1)      # '#111921'
COLOR_WHITE = stg_color('white', depth=1)      # '#FFFFFF'

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Bilingual text dictionary
TEXTS = {
    'title': {
        'English': 'Copper Plating Consumption Optimizer',
        '中文': '镀铜工艺优化器'
    },
    'subtitle': {
        'English': 'AI-powered optimization for pattern plating',
        '中文': '基于人工智能的图形电镀工艺优化'
    },
    'company': {
        'English': 'JST · STARTEAM GLOBAL',
        '中文': 'JST · 先特集团'
    },
    'enter_specifications': {
        'English': 'Enter Board Specifications',
        '中文': '输入板件参数'
    },
    'board_specifications': {
        'English': 'Board Specifications',
        '中文': '板件规格'
    },
    'quality_requirements': {
        'English': 'Quality Requirements',
        '中文': '质量要求'
    },
    'line_type': {
        'English': 'Plating Line Type',
        '中文': '电镀线类型'
    },
    'vcp_line': {
        'English': 'VCP Line',
        '中文': 'VCP线'
    },
    'non_vcp_line': {
        'English': 'Non-VCP Line',
        '中文': '非VCP线'
    },
    'board_thickness': {
        'English': 'Board Thickness',
        '中文': '板厚'
    },
    'unit_mm': {
        'English': 'Unit: millimeters (mm)',
        '中文': '单位：毫米 (mm)'
    },
    'aspect_ratio': {
        'English': 'Aspect Ratio',
        '中文': '纵横比'
    },
    'aspect_ratio_formula': {
        'English': 'Formula: board thickness / hole diameter',
        '中文': '公式：板厚 / 孔径'
    },
    'required_thickness': {
        'English': 'Customer Required Thickness',
        '中文': '客户要求铜厚'
    },
    'unit_micrometer': {
        'English': 'Unit: micrometers (μm)',
        '中文': '单位：微米 (μm)'
    },
    'calculate_button': {
        'English': 'Calculate Optimal Parameters',
        '中文': '计算最优参数'
    },
    'calculating': {
        'English': 'Computing optimal parameters...',
        '中文': '正在计算最优参数...'
    },
    'recommended_parameters': {
        'English': 'Recommended Parameters',
        '中文': '推荐参数'
    },
    'plating_time': {
        'English': 'Plating Time',
        '中文': '电镀时间'
    },
    'minutes': {
        'English': 'minutes',
        '中文': '分钟'
    },
    'current_density': {
        'English': 'Current Density',
        '中文': '电流密度'
    },
    'unit_a_cm2': {
        'English': 'A/cm²',
        '中文': '安培/平方厘米'
    },
    'spray_frequency': {
        'English': 'Spray Frequency',
        '中文': '喷淋频率'
    },
    'unit_hz': {
        'English': 'Hertz (Hz)',
        '中文': '赫兹 (Hz)'
    },
    'line_type_non_vcp': {
        'English': 'Non-VCP',
        '中文': '非VCP线'
    },
    'process_summary': {
        'English': 'Process Summary',
        '中文': '工艺摘要'
    },
    'input_parameters': {
        'English': 'Input Parameters',
        '中文': '输入参数'
    },
    'output_parameters': {
        'English': 'Output Parameters',
        '中文': '输出参数'
    },
    'download_results': {
        'English': 'Export Results (CSV)',
        '中文': '导出结果 (CSV)'
    },
    'calculation_error': {
        'English': 'Calculation error',
        '中文': '计算错误'
    },
    'technical_details': {
        'English': 'Technical details',
        '中文': '技术详情'
    },
    'footer': {
        'English': 'JST Copper Plating Process Optimizer | Version 1.0',
        '中文': 'JST镀铜工艺优化器 | 版本 1.0'
    },
    'powered_by': {
        'English': 'Powered by STARTEAM GLOBAL Data Insights Team',
        '中文': '由数据科学团队提供支持'
    },
    'model_not_available': {
        'English': 'Model Not Available',
        '中文': '模型不可用'
    },
    'ensure_files': {
        'English': 'Please ensure training data and config files are in place.',
        '中文': '请确保训练数据和配置文件已就位。'
    }
}

def get_text(key):
    """Get text in selected language"""
    return TEXTS[key][st.session_state.language]

# Page configuration
st.set_page_config(
    page_title="JST Copper Plating Consumption Optimizer",
    page_icon="",
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

# Professional CSS - All Blue Theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {COLOR_GRAY_4}30 0%, {COLOR_WHITE} 100%);
    }}
    
    .main-header {{
        font-size: 2.5rem;
        color: {COLOR_BLUE_1};
        text-align: center;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }}
    
    .result-card {{
        background: {COLOR_WHITE};
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid {COLOR_GRAY_4};
        transition: all 0.2s ease;
    }}
    
    .result-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
    
    .info-box {{
        background: {COLOR_WHITE};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 3px solid {COLOR_BLUE_2};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
        transition: all 0.2s ease;
        width: 100%;
    }}
    
    .stButton > button:hover {{
        background: {COLOR_BLUE_2};
        transform: translateY(-1px);
    }}
    
    .stDownloadButton > button {{
        background: {COLOR_GRAY_1};
        color: {COLOR_WHITE} !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: {COLOR_BLUE_1};
    }}
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {{
        border-radius: 4px;
        border: 1px solid {COLOR_GRAY_4};
        transition: all 0.2s ease;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {{
        border-color: {COLOR_BLUE_1};
        box-shadow: 0 0 0 2px {COLOR_BLUE_4};
    }}
    
    .stSlider > div > div > div {{
        background-color: {COLOR_BLUE_4};
    }}
    
    .stSlider > div > div > div > div {{
        background-color: {COLOR_BLUE_2};
    }}
    
    hr {{
        border: none;
        height: 1px;
        background: {COLOR_GRAY_4};
        margin: 1.5rem 0;
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
        margin-top: 0.2rem;
    }}
    
    .info-table {{
        width: 100%;
        font-size: 0.85rem;
    }}
    
    .info-table td {{
        padding: 4px 0;
    }}
    
    .info-table td:first-child {{
        color: {COLOR_GRAY_1};
    }}
    
    .info-table td:last-child {{
        font-weight: 500;
        color: {COLOR_BLUE_1};
    }}
</style>
""", unsafe_allow_html=True)

# Header with logo in top-left corner
col_logo, col_empty = st.columns([1, 5])

with col_logo:
    if logo_exists:
        # Open and resize with high quality
        img = Image.open(logo_path)
        # Resize while maintaining aspect ratio
        max_width = 120
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        st.image(img, use_container_width=False)

# Title centered below logo
st.markdown(f'<div class="main-header">{get_text("title")}</div>', unsafe_allow_html=True)

# File paths - use relative paths
TRAINING_DATA_PATH = current_dir / "train_data.csv"
MODEL_CONFIG_PATH = current_dir / "src" / "copper_usage" / "config" / "default_models.yaml"
DATA_CONFIG_PATH = current_dir / "src" / "copper_usage" / "config" / "data_settings.yaml"
MODEL_SAVE_PATH = current_dir / "trained_model.pkl"

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_attempted' not in st.session_state:
    st.session_state.training_attempted = False

# Model training - no UI display
if not st.session_state.model_loaded and PACKAGES_AVAILABLE and not st.session_state.training_attempted:
    st.session_state.training_attempted = True
    
    try:
        # Check if model already exists
        if MODEL_SAVE_PATH.exists():
            with open(MODEL_SAVE_PATH, 'rb') as f:
                st.session_state.model = pickle.load(f)
                st.session_state.model_loaded = True
        else:
            # Check if training files exist
            files_exist = all([
                TRAINING_DATA_PATH.exists(),
                MODEL_CONFIG_PATH.exists(),
                DATA_CONFIG_PATH.exists()
            ])
            
            if files_exist:
                # Silent training - no progress bar or messages
                model = ModelFactory.build_model_from_config(str(MODEL_CONFIG_PATH))
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
        pass

# Check if model is loaded - show warning only if not loaded and files exist
if not st.session_state.model_loaded:
    # Only show warning if files exist but model failed to load
    if TRAINING_DATA_PATH.exists() and MODEL_CONFIG_PATH.exists():
        st.warning(f"""
        ### {get_text('model_not_available')}
        
        {get_text('ensure_files')}
        """)
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
            get_text('board_thickness'),
            min_value=0.1,
            max_value=10.0,
            value=1.1,
            step=0.1,
            format="%.2f"
        )
        st.markdown(f'<div class="caption">{get_text("unit_micrometer")}</div>', unsafe_allow_html=True)
        
        ratio = st.number_input(
            get_text('aspect_ratio'),
            min_value=1.0,
            max_value=20.0,
            value=4.5,
            step=0.1,
            format="%.2f"
        )
        st.markdown(f'<div class="caption">{get_text("aspect_ratio_formula")}</div>', unsafe_allow_html=True)
        
        # Add spacing at bottom of left column
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**{get_text('quality_requirements')}**")
        
        minimal_thickness = st.number_input(
            get_text('required_thickness'),
            min_value=5.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
            format="%.2f"
        )
        st.markdown(f'<div class="caption">{get_text("unit_micrometer")}</div>', unsafe_allow_html=True)
        
        # Add spacing before the button
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Add spacing before button row
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Centered button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submitted = st.form_submit_button(get_text('calculate_button'), type="primary", use_container_width=True)
    
    # Process form submission with fixed margin (5%)
    if submitted:
        try:
            with st.spinner(get_text('calculating')):
                board = BoardFeatureContainer(
                    is_vcp=is_vcp,
                    Ratio=ratio,
                    board_thickness=board_thickness,
                    minimal_thickness=minimal_thickness,
                    margin=FIXED_MARGIN  # Fixed at 5%
                )
                
                result = st.session_state.model.predict(board)
                
                st.session_state.calculation_result = {
                    'result': result,
                    'is_vcp': is_vcp,
                    'board_thickness': board_thickness,
                    'ratio': ratio,
                    'minimal_thickness': minimal_thickness,
                    'margin': FIXED_MARGIN
                }
                
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
    minimal_thickness = data['minimal_thickness']
    
    st.markdown("---")
    st.markdown(f"## {get_text('recommended_parameters')}")
    
    # Create three columns for the parameter cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Plating Time Card
        st.markdown(f"""
        <div style="background: {COLOR_WHITE}; border-radius: 12px; padding: 1.2rem; margin: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid {COLOR_GRAY_4}; text-align: center;">
            <div style="font-size: 0.75rem; color: {COLOR_GRAY_2}; text-transform: uppercase; letter-spacing: 0.5px;">{get_text('plating_time')}</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {COLOR_BLUE_1}; margin: 0.5rem 0;">{result.plating_time:.1f}</div>
            <div style="font-size: 0.8rem; color: {COLOR_GRAY_1};">{get_text('minutes')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Current Density Card
        st.markdown(f"""
        <div style="background: {COLOR_WHITE}; border-radius: 12px; padding: 1.2rem; margin: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid {COLOR_GRAY_4}; text-align: center;">
            <div style="font-size: 0.75rem; color: {COLOR_GRAY_2}; text-transform: uppercase; letter-spacing: 0.5px;">{get_text('current_density')}</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {COLOR_BLUE_1}; margin: 0.5rem 0;">{result.current_density:.2f}</div>
            <div style="font-size: 0.8rem; color: {COLOR_GRAY_1};">A/cm²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Spray Frequency or Line Type Card
        if is_vcp and hasattr(result, 'spray_frequency') and result.spray_frequency is not None:
            st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border-radius: 12px; padding: 1.2rem; margin: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid {COLOR_GRAY_4}; text-align: center;">
                <div style="font-size: 0.75rem; color: {COLOR_GRAY_2}; text-transform: uppercase; letter-spacing: 0.5px;">{get_text('spray_frequency')}</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {COLOR_BLUE_1}; margin: 0.5rem 0;">{result.spray_frequency:.1f}</div>
                <div style="font-size: 0.8rem; color: {COLOR_GRAY_1};">Hz</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border-radius: 12px; padding: 1.2rem; margin: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid {COLOR_GRAY_4}; text-align: center;">
                <div style="font-size: 0.75rem; color: {COLOR_GRAY_2}; text-transform: uppercase; letter-spacing: 0.5px;">{get_text('line_type')}</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: {COLOR_BLUE_1}; margin: 0.5rem 0;">{get_text('non_vcp_line')}</div>
                <div style="font-size: 0.8rem; color: {COLOR_GRAY_1};">No spray control</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary section - both input and output boxes
    st.markdown("---")
    st.markdown(f"### {get_text('process_summary')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: {COLOR_WHITE}; border-radius: 10px; padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid {COLOR_BLUE_2}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem; color: {COLOR_BLUE_1};"> {get_text('input_parameters')}</div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                <span style="color: {COLOR_GRAY_1};">{get_text('line_type')}</span>
                <span style="font-weight: 500;">{get_text('vcp_line') if is_vcp else get_text('non_vcp_line')}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                <span style="color: {COLOR_GRAY_1};">{get_text('board_thickness')}</span>
                <span style="font-weight: 500;">{board_thickness} mm</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                <span style="color: {COLOR_GRAY_1};">{get_text('aspect_ratio')}</span>
                <span style="font-weight: 500;">{ratio:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0;">
                <span style="color: {COLOR_GRAY_1};">{get_text('required_thickness')}</span>
                <span style="font-weight: 500;">{minimal_thickness} μm</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Build output parameters list
        if is_vcp and hasattr(result, 'spray_frequency') and result.spray_frequency is not None:
            st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border-radius: 10px; padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid {COLOR_BLUE_2}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem; color: {COLOR_BLUE_1};"> {get_text('output_parameters')}</div>
                <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                    <span style="color: {COLOR_GRAY_1};">{get_text('plating_time')}</span>
                    <span style="font-weight: 500; color: {COLOR_BLUE_1};">{result.plating_time:.1f} {get_text('minutes')}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                    <span style="color: {COLOR_GRAY_1};">{get_text('current_density')}</span>
                    <span style="font-weight: 500; color: {COLOR_BLUE_1};">{result.current_density:.2f} A/cm²</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.4rem 0;">
                    <span style="color: {COLOR_GRAY_1};">{get_text('spray_frequency')}</span>
                    <span style="font-weight: 500; color: {COLOR_BLUE_1};">{result.spray_frequency:.1f} Hz</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border-radius: 10px; padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid {COLOR_BLUE_2}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem; color: {COLOR_BLUE_1};"> {get_text('output_parameters')}</div>
                <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid {COLOR_GRAY_4};">
                    <span style="color: {COLOR_GRAY_1};">{get_text('plating_time')}</span>
                    <span style="font-weight: 500; color: {COLOR_BLUE_1};">{result.plating_time:.1f} {get_text('minutes')}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.4rem 0;">
                    <span style="color: {COLOR_GRAY_1};">{get_text('current_density')}</span>
                    <span style="font-weight: 500; color: {COLOR_BLUE_1};">{result.current_density:.2f} A/cm²</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Download button
    export_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_vcp": is_vcp,
        "Ratio": ratio,
        "board_thickness_mm": board_thickness,
        "minimal_thickness_um": minimal_thickness,
        "margin": FIXED_MARGIN,
        "plating_time_min": result.plating_time,
        "current_density_A_cm2": result.current_density,
    }
    
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