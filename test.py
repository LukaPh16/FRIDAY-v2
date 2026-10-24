def print_red_box(text):
    print("\033[91m{}\033[0m".format(text))

print_red_box("Hello World")