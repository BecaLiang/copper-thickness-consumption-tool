import os
import json
from hashlib import sha256

from pathlib import Path

from typing import Any

from functools import wraps, partial
from collections import namedtuple
from time import time
from datetime import date, datetime


TimeDeltaTuple = namedtuple('timedelta', ('days', 'hours', 'minutes', 'seconds'))


weekdays = {
    1: 'Monday',
    2: 'Tuesday', 
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday',
    7: 'Sunday',
}


def make_int(
        value: Any, 
        raise_value_error: bool=False, 
        alternative: Any=None
    ) -> int | None:
    # use pd.NA as alternative in case of DataFrames
    try:
        int_value = int(value)
    except ValueError:
        if raise_value_error:
            raise ValueError
        else:
            return alternative
    except TypeError:
        return alternative
    if int_value == float(value):
        return int_value
    else:
        if raise_value_error:
            raise ValueError
        else:
            return alternative
    

def break_down_seconds_to_periods(duration: float) -> TimeDeltaTuple:
    seconds = duration % 60
    rest = duration // 60
    minutes = rest % 60
    rest = rest // 60
    hours = rest % 24
    days = rest // 24
    return TimeDeltaTuple(days, hours, minutes, seconds)


def break_down_seconds_to_string(duration: float, second_precision=3) -> str:
    duration_tuple = break_down_seconds_to_periods(duration)
    skip_zero = True
    msg_string = ""
    for k, v in duration_tuple._asdict().items():
        if v == 0 and skip_zero:
            continue
        precision = second_precision if k == 'seconds' else 0
        msg_string += f"{v:.{precision}f} {k}, "
        skip_zero = False
    return msg_string.strip(', ')


def get_last_week_of_year(year: int) -> int:
    return date(year, 12, 28).isocalendar().week


def dict_to_hash(this_dict: dict) -> str:
    return sha256(json.dumps(this_dict).encode('utf-8')).hexdigest()


def sort_dict_by_value(the_dict: dict, reverse: bool=False):
    return dict(
        sorted(
            the_dict.items(), 
            key=lambda item: item[1], 
            reverse=reverse,
        )
    )


def dt_to_string(pit: datetime=None) -> str:
    dtp = pit or datetime.now()
    seconds = int(dtp.hour * 60 * 60 + dtp.minute * 60 + dtp.second)
    return f'{dtp.year}-{dtp.month:02d}-{dtp.day:02d}-{seconds}'


def dt_from_string(dt_str: str) -> datetime:
    assert dt_str.count('_') == dt_str.count('.') == 0
    parts = dt_str.split('-')
    assert len(parts) == 4
    ttdd = break_down_seconds_to_periods(int(parts[-1]))
    return datetime.fromisoformat(
        f'{parts[0]}-{parts[1]}-{parts[2]} {ttdd.hours:02d}:{ttdd.minutes:02d}:{ttdd.seconds:02d}'
    )


def time_function_call(f):
    """simply prints name of the function with standard formatting"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f'executing {f.__name__} took {te-ts:.2f} seconds')
        return result
    return wrapper


def time_function_call_custom_msg(message, convert_seconds=True, second_precision=3):
    def timer_decorator(f):
        def wrapper(*args, **kwargs):
            ts = time()
            result = f(*args, **kwargs)
            te = time()
            if convert_seconds:
                dur_str_s = break_down_seconds_to_string(
                    te-ts, second_precision=second_precision
                )
                print(
                    f'{message} took {dur_str_s}'
                )
            else:
                print(f'{message} took {te-ts:.{second_precision}f} seconds')
            return result
        return wrapper
    return timer_decorator


def get_module_sub_path(
    module_path: str, 
    subpath_string: str,
    max_depth: int=10,
):
    assert max_depth > 0 and isinstance(max_depth, int)
    parent_path = cfg_path = module_path
    counter = 0
    while not os.path.isdir(cfg_path):
        if counter == max_depth:
            raise FileNotFoundError(
                f'Max recursion depth reached for {subpath_string} in {module_path}'
            )
        parent_path = os.path.split(parent_path)[0]
        cfg_path = os.path.join(parent_path, subpath_string)
        counter += 1
    return cfg_path


def get_module_base_path(
        py_file: str, 
        marker_file: str='setup.py', 
        max_iterations: int=10,
    ):

    if os.path.isdir(py_file):
        parent_path = py_file
    elif os.path.isfile(py_file):
        parent_path = os.path.split(py_file)[0]
    else:
        raise TypeError(f'{py_file} should either be a file or a directory')

    counter = 1
    while not marker_file in os.listdir(parent_path):
        parent_path = os.path.split(parent_path)[0]
        counter += 1
        if counter > max_iterations:
            raise FileNotFoundError

    return parent_path


find_config_path = partial(get_module_sub_path, subpath_string='config')


def can_be_path(obj):
    if isinstance(obj, str):
        return True
    elif isinstance(obj, bytes):
        return True
    elif isinstance(obj, os.PathLike):
        return True
    return False


def get_config_file(
        file_name: str,
        explicit_path: str='',
        implicit_path: str='',
        module_path: str=None,
):
    
    cfg_file_candidate = os.path.join(explicit_path or '', file_name)
    if os.path.isfile(cfg_file_candidate):
        return cfg_file_candidate
    cfg_file_candidate = os.path.join(implicit_path or '', file_name)
    if os.path.isfile(cfg_file_candidate):
        return cfg_file_candidate
    if module_path is not None:
        cfg_file_candidate = os.path.join(
            find_config_path(module_path), 
            file_name,
        )
    if os.path.isfile(cfg_file_candidate):
        return cfg_file_candidate
    raise FileNotFoundError(f'Could not find {file_name}')


def numpy_free_frange(start, stop, step):
    import decimal
    x = decimal.Decimal(start)
    step = decimal.Decimal(step)
    while x < stop:
        yield float(x)
        x += decimal.Decimal(step)


def expand_ranged_parameter(param: str | list, slicer: str=':'):
    if isinstance(param, list):
        return param
    elif isinstance(param, str):
        pparts = param.split(slicer)
        match len(pparts):
            case 1:
                # raise warning?
                return param
            case 2:
                try:
                    return list(
                        range(
                            int(pparts[0]),
                            int(pparts[1]),
                        )
                    )
                except ValueError:
                    raise ValueError('for float parameters, pass step-size')
            case 3:
                try:
                    start=int(pparts[0])
                    end=int(pparts[1])
                    step=int(pparts[2])
                    return list(
                        range(
                            start, end, step
                        )                        
                    )
                except ValueError:
                    start=float(pparts[0])
                    end=float(pparts[1])
                    step=float(pparts[2])
                    try:
                        import numpy as np
                        return list(
                            np.arange(
                                start, end, step
                            )
                        )
                    except ImportError:
                        return list(
                            numpy_free_frange(
                                start=start,
                                stop=end,
                                step=step,
                            )
                        )
            case _:
                raise ValueError(f'Too many {slicer} in ts-param-parser')
    else:
        return param


def is_this_fabric():
    return Path('/lakehouse').exists()