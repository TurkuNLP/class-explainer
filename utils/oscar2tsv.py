import os
import gzip
import json
import sys

lang = sys.argv[1]#'fr'
PATH = f'../../anna/oscar/data/{lang}/linewise/'
#PATH = f'../../anna/oscar/data/{lang}/1/'

c = 0
with open(f"oscar_data_{lang}.tsv", 'w', encoding='utf-8') as outf:
	print("Reading", PATH)
	for dir in sorted(os.listdir(PATH)):
		#for dir in [PATH]:
		subdir = os.path.join(PATH, dir)
		#subdir = PATH
		for fn in sorted(os.listdir(subdir)):
			if not fn.endswith('.gz'):
				continue
			print(subdir, fn, c)
			for row in gzip.open(os.path.join(subdir, fn), 'rt', encoding='utf-8'):
				data = json.loads(row)
				try:
					text = ' '.join([t for t,l in zip(data['text'], data['linelabels']) if l[0] == 'ok'])
				except:
					text = data['text'] # Not linewise
				labels = [l for l in data['labels'] if l]
				if len(labels) != 1:
					continue
				print('%s\t%s' % (labels[0], text), file=outf)
				c += 1


#{"id": "1", "labels": ["IN"], "linelabels": [["ok", 0.72691], ["ok", 0.91672]], "text": ["24 janv. 2018 Sources de donn<C3><A9>esData Sources. Les types de donn<C3><A9>es sont organis<C3><A9>s dans les cat<C3><A9>gories suivantes :Data types are organized in the following categories: ToutesAll; FichierFile; Base de donn<C3><A9>esDatabase; AzureAzure; Online ServicesOnline Services; AutreOther. La cat<C3><A9>gorie Toutes comprend tous les . Bring up to date in french", "Description. Cr<C3><A9>ez un agenda d'<C3>
#<A9>v<C3><A8>nements et g<C3><A9>rez le facilement. L'extension The Events Calendar fournit une qualit<C3><A9> de niveau professionnel et des fonctionnalit<C3><A9>s <C3><A9>prouv<C3><A9>es par une <C3><A9>quipe en qui vous pouvez avoir confiance. Empaquet<C3><A9> avec plein de fonctionalit<C3><A9>s, The Event Calendar de Modern Tribe est pr<C3><AA>t d<C3><A8>s . Bring up to date in french"]}
#/users/ronnqvis/varieties/anna/oscar/data/fr/linewise/1
