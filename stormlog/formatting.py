"""Framework-independent byte formatting."""


def format_bytes(bytes_value: int, precision: int = 2) -> str:
    """
    Format bytes into human-readable format.

    Args:
        bytes_value: Number of bytes
        precision: Decimal precision

    Returns:
        Formatted string (e.g., "1.25 GB")
    """
    if bytes_value == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    sign = "-" if bytes_value < 0 else ""
    size = float(abs(bytes_value))

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{sign}{size:.{precision}f} {units[unit_index]}"
