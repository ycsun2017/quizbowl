#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:59:34 2018

@author: yanchaosun
"""


import os
import time
import torch
import argparse
import logging
import json

from tqdm import tqdm
from drqa.reader import Predictor
from drqa import DATA_DIR as DRQA_DATA

DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = os.path.join(DRQA_DATA, 'reader')
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, default=MODEL_DIR,
                   help='Directory for saved models')
parser.add_argument('--model-name', type=str, default='first_train.mdl',
                   help='Unique model identifier')
parser.add_argument('--test-dir', type=str, default=DATA_DIR,
                   help='Directory of data to be evaluated')
parser.add_argument('--test-file', type=str,
                   default='test.json',
                   help='name of data to be evaluated')
parser.add_argument('--embedding-file', type=str, default='glove.840B.300d.txt',
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default=DATA_DIR,
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default='regexp',
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')

args = parser.parse_args()
t0 = time.time()

args.test_file = os.path.join(args.test_dir, args.test_file)
if not os.path.isfile(args.test_file):
    raise IOError('No such file: %s' % args.test_file)
args.model_name = os.path.join(args.model_dir, args.model_name)
if not os.path.isfile(args.model_name):
    raise IOError('No such file: %s' % args.model_name)
args.embedding_file = os.path.join(EMBED_DIR, args.embedding_file)
if not os.path.isfile(args.embedding_file):
    raise IOError('No such file: %s' % args.embedding_file)


args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')




# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------


examples = []
qids = []
answers = []
#contexts = []
#ques = []
#with open(args.test_file) as f:
#    data = json.load(f)['data']
#    for article in data:
#        for paragraph in article['paragraphs']:
#            context = paragraph['context']
#            for qa in paragraph['qas']:
#                qids.append(qa['id'])
#                examples.append((context, qa['question']))
#                if 'answers' in qa:
#                    ans.append(qa['answers'][0]['text'])
#                else:
#                    ans.append(" ")

with open(args.test_file, 'r') as f:
        data = json.load(f)
        questions = data['questions']
        n = len(questions)
        print("loading",n,"questions")
        
        for i in range(n):
            question = questions[i]
            ans = question["page"].replace("_"," ")
            answers.append(ans)
            paragraphs = []
            for article in question['annotated_paras']:
                for paras in article:
                    context = paras["paragraph"]
#                    contexts.append(context)
#                    ques.append(question["text"])
                    qids.append(i)
                    examples.append((context, question["first_sentence"]))
#print("examples",len(examples),examples[0])
#print("qids", len(qids),qids[0]) 
#print("answers",len(answers),answers[0])   

predictor = Predictor(
    model=args.model_name,
    tokenizer=args.tokenizer,
    embedding_file=args.embedding_file,
    num_workers=args.num_workers,
)
if args.cuda:
    predictor.cuda()
      
results = {}
for i in tqdm(range(0, len(examples), args.batch_size)):
    predictions = predictor.predict_batch(
        examples[i:i + args.batch_size], top_n=1
    )
#    print(predictions)
    for j in range(len(predictions)):
        result = [(p[0], float(p[1])) for p in predictions[j]]
        qid = qids[i+j]
        if qid not in results.keys():
            results[qid] = []
        results[qid].append(result)
#        print("append", qid, result)
    
#print(len(results))
#print(results[0])

outputs = {}
right_sum = 0
total_sum = n
for q in results.keys():
    max_v = 0
    max_res = None
    for res in results[q]:
        for r in res:
            if r[1] > max_v:
                max_v = r[1]
                max_res = r
    outputs[q] = max_res
    if max_res[0] == answers[q]:
        right_sum += 1
#print(outputs[0])
print("right_sum", right_sum)
print("acc", float(right_sum) / total_sum)

#results = {}
#for i in tqdm(range(0, len(examples), args.batch_size)):
#    predictions = predictor.predict_batch(
#        examples[i:i + args.batch_size], top_n=args.top_n
#    )
#    for j in range(len(predictions)):
#        results[qids[i + j]] = [(p[0], float(p[1]), ans[i+j]) for p in predictions[j]]
#    if i > 1:
#        break

model = os.path.splitext(os.path.basename(args.model_name or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.test_file))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '.preds')

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    json.dump(outputs, f)

logger.info('Total time: %.2f' % (time.time() - t0))
