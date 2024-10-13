
def convert_milliseconds(milliseconds):
    seconds = milliseconds // 1000
    r_milliseconds = milliseconds % 1000
    minutes = seconds // 60
    r_seconds = seconds % 60
    hours = minutes // 60
    r_minutes = minutes % 60
    days = hours // 24
    r_hours = hours % 24
    result = ""

    if days != 0:
        result += f"{int(days)}d "
    if r_hours != 0:
        result += f"{int(r_hours)}h "
    if r_minutes != 0:
        result += f"{int(r_minutes)}m "
    if r_seconds != 0:
        result += f"{int(r_seconds)}s "
    if r_milliseconds != 0:
        result += f"{r_milliseconds}ms"

    return result

