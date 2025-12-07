import random


def generate_wifi_pass():
    num = random.randint(0, 999)
    return f"Hilton!Guest{num:03d}"
