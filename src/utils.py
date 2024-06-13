import datetime


def get_current_date_as_string():
    """Returns the current date formatted as 'yyyymmdd'."""
    return datetime.datetime.now().strftime("%Y%m%d")
