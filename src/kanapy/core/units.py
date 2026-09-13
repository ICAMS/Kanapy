"""Length units for output; all internal lengths are in micrometers."""


def normalize_length_unit(unit: str) -> str:
    """Validate an output unit and normalize the micro sign to ASCII ``um``."""
    if not isinstance(unit, str) or unit not in ('µm', 'um', 'mm', 'm'):
        raise ValueError(f'Output units must be "µm", "um", "mm", or "m", not {unit!r}.')
    return 'um' if unit == 'µm' else unit


def length_scale_from_um(unit: str) -> float:
    """Return the factor converting internal micrometers to the output unit."""
    return {'um': 1.0, 'mm': 1e-3, 'm': 1e-6}[normalize_length_unit(unit)]
