import re
#import nltk
#from nltk import sent_tokenize
#import string
# - divisione dei testi per frasi
# - togliere punteggiatura

def removeTags(path_file):
	text=open(path_file,"r").read()
	text_r = re.sub('<META(.*)>(\n.*)*<\/teiHeader>|<head(.*)>(.*)<\/head>|<app(.*)>(.*)<\/app>|<foreign(.*)>(.*)<\/foreign>|<quote(.*)>(.*)<\/quote>|<argument().*>(.*\n)*<\/p>|<note(.*)>(.*)<\/note>|<rf(.*)>(.*)<\/rf>|<i(.*)>(.*)<\/i>|<[^<]+>', "", text)
	with open(path_file, "w") as f:
		f.write(text_r)

  		
#def divide_sent(path_file):
#	text = open(path_file,"r").read()
#	sent = nltk.sent_tokenize(text)


#def removePunct(path_file):
	#text = open(path_file, "r").read()
	#text_p = "".join([char for char in text if char not in string.punctuation])
	#with open(path_file,"w") as f:
		#f.write(text_p)
	

    # def divide_fragments(path_file):
#     text = open(path_file, "r").read()
#     text = text.replace("\n", " ")
#     fragments = re.findall(r'<div2(.*?)*>(.*?)<\/div2>', text)
#     if not fragments:
#         print(path_file)
