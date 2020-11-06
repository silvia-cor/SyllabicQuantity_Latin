import re

# ? studio xml tags
# - togliere le cose da togliere (header, citazioni, tags..)
# - divisione dei testi per frasi
# - togliere punteggiatura
#
#
# def divide_fragments(path_file):
#     text = open(path_file, "r").read()
#     text = text.replace("\n", " ")
#     fragments = re.findall(r'<div2(.*?)*>(.*?)<\/div2>', text)
#     if not fragments:
#         print(path_file)