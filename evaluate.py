import os
import json
import time
import click
import pickle
import signal
import requests
import subprocess
import numpy as np
import logging
import socket
import errno
from tqdm import tqdm


elog = logging.getLogger('eval')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('evaluation.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logging.getLogger('requests').setLevel(logging.CRITICAL)


class CurveScore:
    def __init__(self, curve_pkl='../curve_pipeline.pkl'):
        with open(curve_pkl, 'rb') as f:
            self.pipeline = pickle.load(f)

    def get_weight(self, x):
        return self.pipeline.predict(np.asarray([[x]]))[0]

    def score(self, guesses, question):
        '''guesses is a list of {'guess': GUESS, 'buzz': True/False}
        '''
        char_length = len(question['text'])
        buzzes = [x['buzz'] for x in guesses]
        if True not in buzzes:
            return 0
        buzz_index = buzzes.index(True)
        rel_position = guesses[buzz_index]['char_index'] / char_length
        weight = self.get_weight(rel_position)
        result = guesses[buzz_index]['guess'] == question['page']
        return weight * result


def start_server():
    web_proc = subprocess.Popen(
        'bash run.sh', shell=True,
        preexec_fn=os.setsid
    )
    return web_proc


def retry_get_url(url, retries=5, delay=3):
    while retries > 0:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.ConnectionError as e:
            retries -= 1
            elog.warn(e)

        if delay > 0:
            time.sleep(delay)
    return None


def get_question_query(qid, question, char_idx):
    contexts = []
    for article in question['annotated_paras']:
        for paras in article:
            contexts.append(paras["paragraph"])
    char_idx = min(char_idx, len(question['text']))
    for sent_idx, (st, ed) in enumerate(question['tokenizations']):
        if char_idx >= st and char_idx <= ed:
            break
        
    query = {
            'question_idx': qid,
            'sent_index': sent_idx,
            'char_index': char_idx,
            'text': question['text'][:char_idx],
            'contexts': contexts
    }
#    print(query)
    return query


def get_answer_single(url, questions, char_step_size):
    elog.info('Collecting responses to questions')
    answers = []
    for question_idx, q in enumerate(tqdm(questions)):
        elog.info(f'Running question_idx={question_idx} qnum={q["qanta_id"]}')
        answers.append([])
        # get an answer every K characters
        for char_idx in range(1, len(q['text']) + char_step_size,
                              char_step_size):
            query = get_question_query(question_idx, q, char_idx)
            resp = requests.post(url, json=query).json()
            query.update(resp)
            query.pop("contexts")
            answers[-1].append(query)
    return answers


def get_answer_batch(url, questions, char_step_size, batch_size):
    elog.info('Collecting responses to questions in batches', batch_size)
    answers = []
    batch_ids = list(range(0, len(questions), batch_size))
    for batch_idx in tqdm(batch_ids):
        batch_ed = min(len(questions), batch_idx + batch_size)
        qs = questions[batch_idx: batch_ed]
        max_len = max(len(q['text']) for q in qs)
        qids = list(range(batch_idx, batch_ed))
        answers += [[] for _ in qs]
        for char_idx in range(1, max_len + char_step_size, char_step_size):
            query = {'questions': []}
            for i, q in enumerate(qs):
                query['questions'].append(
                    get_question_query(qids[i], q, char_idx))
            resp_raw = requests.post(url, json=query)
            resp = resp_raw.json()
            for i, r in enumerate(resp):
                q = query['questions'][i]
                q.update(r)
                q.pop("contexts")
                answers[qids[i]].append(q)
    return answers


def check_port(hostname, port):
    pass


@click.command()
@click.argument('input_dir')
@click.argument('output_dir', default='predictions.json')
@click.argument('score_dir', default='scores.json')
@click.option('--char_step_size', default=25)
@click.option('--hostname', default='0.0.0.0')
@click.option('--norun-web', default=False, is_flag=True)
@click.option('--wait', default=0, type=int)
@click.option('--curve-pkl', default='curve_pipeline.pkl')
def evaluate(input_dir, output_dir, score_dir, char_step_size, hostname,
             norun_web, wait, curve_pkl):
    try:
        if not norun_web:
            web_proc = start_server()

        if wait > 0:
            time.sleep(wait)

        status_url = f'http://{hostname}:4861/api/1.0/quizbowl/status'
        status = retry_get_url(status_url)
        elog.info(f'API Status: {status}')
        if status is None:
            elog.warning('Failed to find a running web server beep boop, prepare for RUD')
            raise ValueError('Status API could not be reached')

        with open(input_dir) as f:
            questions = json.load(f)['questions'][:40]
        if status is not None and status['batch'] is True:
            url = f'http://{hostname}:4861/api/1.0/quizbowl/batch_act'
            answers = get_answer_batch(url, questions,
                                       char_step_size,
                                       status['batch_size'])
        else:
            url = f'http://{hostname}:4861/api/1.0/quizbowl/act'
            answers = get_answer_single(url, questions,
                                        char_step_size)

        with open(output_dir, 'w') as f:
            json.dump(answers, f)

        elog.info('Computing curve score of results')
        curve_score = CurveScore(curve_pkl=curve_pkl)
        sent1_results = []
        eoq_results = []
        curve_results = []
        for question_idx, guesses in enumerate(answers):
            question = questions[question_idx]
            sent1_guess = None
            for g in guesses:
                if g['sent_index'] == 1:
                    sent1_guess = g['guess']
                    break
            sent1_results.append(sent1_guess == question['page'])
            eoq_results.append(guesses[-1]['guess'] == question['page'])
            curve_results.append(curve_score.score(guesses, question))
        eval_out = {
            'sent1_acc': sum(sent1_results) * 1.0 / len(sent1_results),
            'eoq_acc': sum(eoq_results) * 1.0 / len(eoq_results),
            'curve': sum(curve_results) * 1.0 / len(curve_results),
        }
        with open(score_dir, 'w') as f:
            json.dump(eval_out, f)
        print(json.dumps(eval_out))

    finally:
        if not norun_web:
            os.killpg(os.getpgid(web_proc.pid), signal.SIGTERM)


if __name__ == '__main__':
    evaluate()
