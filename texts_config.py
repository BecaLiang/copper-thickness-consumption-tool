# texts_config.py
# Bilingual text dictionary for the Streamlit app

TEXTS = {
    'title': {
        'English': 'Copper Plating Consumption Optimizer',
        '中文': '镀铜工艺优化器'
    },
    'enter_specifications': {
        'English': 'Enter Board Specifications',
        '中文': '输入板件参数'
    },
    'select_requirement': {
        'English': 'Select requirement',
        '中文': '选择要求'
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
    'current_density_range': {
        'English': 'Current Density Range',
        '中文': '电流范围'
    },
    'target_thickness': {
        'English': 'Expected Min Thickness',
        '中文': '预期最小铜厚'
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
        'English': 'JST Copper Plating Process Optimizer | Version 2.0',
        '中文': 'JST镀铜工艺优化器 | 版本 2.0'
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

def get_text(key, language):
    """Get text in the specified language"""
    return TEXTS.get(key, {}).get(language, f"Missing: {key}")