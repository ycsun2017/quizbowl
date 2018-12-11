import json
import re

with open("digits.output.json", 'w') as f:
	with open('data/qanta.mapped.2018.04.18.json') as json_data:
		data = json.load(json_data)
		data = data['questions']
		for d in data:
			# index = d['text'].find('For 10')
			# sentence = d['text'][0:index]
			sentence = d['text']
			# digits = [int(s) for s in sentence.split() if s.isdigit()]
			digits = re.findall(r'\d+', sentence)
			digits = [int(digit) for digit in digits if int(digit) > 500]
			if digits is None or len(digits) == 0:
				continue
			d['text'] = digits
			d.pop('tournament', None)
			d.pop('difficulty', None)
			d.pop('proto_id', None)
			d.pop('qdb_id', None)
			d.pop('dataset', None)
			d.pop('tokenizations', None)
			d.pop('first_sentence', None)
			d.pop('gameplay', None)
			d.pop('fold', None)
			print (d)
			f.write(json.dumps(d))
	output = []
	with open('data/qanta.train.2018.04.18.json') as json_data:
		data = json.load(json_data)
		data = data['questions']
		for d in data:
			index = d['text'].find('For 10')
			sentence = d['text'][0:index]
			# digits = [int(s) for s in sentence.split() if s.isdigit()]
			digits = re.findall(r'\d+', sentence)
			digits = [int(digit) for digit in digits if int(digit) > 500]
			if digits is None or len(digits) == 0:
				continue
			d['text'] = digits
			d.pop('tournament', None)
			d.pop('difficulty', None)
			d.pop('proto_id', None)
			d.pop('qdb_id', None)
			d.pop('dataset', None)
			d.pop('tokenizations', None)
			d.pop('first_sentence', None)
			d.pop('gameplay', None)
			d.pop('fold', None)
			print (d)
			output.append(d)
	f.write(json.dumps(output))
