"""Display formatters (currently only the SI-suffix number formatter).

Lives apart from ``stats``/``fileio`` because it is purely a presentation
concern shared by plots and logs.
"""


def human_format(num, decimals=2):
    """Render an integer with SI suffixes (K, M, G, ...).

    Numbers under 10_000 are returned as plain ``str(num)`` (no decimals) since
    the suffixed form would lose precision without any compaction benefit.
    """
    if num < 10000:
        return str(num)

    magnitude = 0
    template = f'%.{decimals}f%s'
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return template % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def human_format_int(x, *args, **kwargs):
    """Matplotlib-compatible formatter (accepts the ``pos`` arg matplotlib passes)."""
    return human_format(int(x), decimals=1)
