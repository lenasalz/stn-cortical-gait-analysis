import contextlib
import os
from typing import Any, Dict, List, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np


def call_silently(func: callable, *args, **kwargs) -> Any:
    """
    Run a function with all stdout/stderr suppressed.
    
    Parameters
    ----------
    func : callable
        The function to run silently.
    *args : tuple
        Positional arguments to pass to func.
    **kwargs : dict
        Keyword arguments to pass to func.
        
    Returns
    -------
    result : Any
        The return value of func.
    """
    with open(os.devnull, "w") as _null, \
         contextlib.redirect_stdout(_null), \
         contextlib.redirect_stderr(_null):
        return func(*args, **kwargs)


def hex_to_rgba(hex_alpha_tuple: Tuple[str, float], default_color: str = 'gray') -> Union[str, Tuple[float, float, float, float]]:
    """
    Convert a (hex, alpha) tuple to an (r, g, b, a) tuple.
    
    Parameters
    ----------
    hex_alpha_tuple : tuple of (str, float) or None
        Tuple containing hex color string and alpha value.
    default_color : str, default 'gray'
        Default color to return if hex_alpha_tuple is None.
        
    Returns
    -------
    color : str or tuple of float
        RGBA color tuple or default color string.
    """
    if hex_alpha_tuple is None:
        return default_color
    hex_val, alpha = hex_alpha_tuple
    return mcolors.to_rgba(hex_val, alpha=alpha)


def generate_color_map(
    groups: List[str],
    blocks: List[str],
    base_colors: List[str],
    baseline_phase: str = 'baseline',
    baseline_alpha: float = 1.0,
    other_alpha: float = 0.3
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """
    Generate a color map for plotting with group-specific colors and phase-specific transparency.
    
    Assigns a base color to each group and varies its transparency for each phase.

    Parameters
    ----------
    groups : list of str
        List of group names (e.g., ['AgeMatched_NFB', 'Patient_NFB']).
    blocks : list of str
        List of condition/phase names (e.g., ['baseline', 'block1']).
    base_colors : list of str
        List of base hex colors, one for each group.
    baseline_phase : str, default 'baseline'
        The name of the baseline phase, which will be fully opaque.
    baseline_alpha : float, default 1.0
        Alpha for baseline phase.
    other_alpha : float, default 0.3
        Alpha for other phases.

    Returns
    -------
    color_map : dict
        A dictionary mapping (group, phase) tuples to (hex color, alpha) tuples.
        
    Raises
    ------
    ValueError
        If not enough base colors are provided for the number of groups.
    """
    color_map = {}
    if len(base_colors) < len(groups):
        raise ValueError("Not enough base colors provided for the number of groups.")

    group_to_base_color = dict(zip(groups, base_colors))

    for group, base_hex in group_to_base_color.items():
        for phase in blocks:
            if phase == baseline_phase:
                color_map[(group, phase)] = (base_hex, baseline_alpha)
            else:
                color_map[(group, phase)] = (base_hex, other_alpha)
    return color_map