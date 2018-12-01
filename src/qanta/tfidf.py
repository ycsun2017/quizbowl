from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import torch
from os import path

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset

from drqa.reader import Predictor

DRQA_MODEL_PATH = 'drqa.mdl'
MODEL_PATH = 'tfidf.pickle'
EMBEDDING_PATH = 'glove.840B.300d.txt'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
DRQA_THRESHOLD = 0.99


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, predictor, questions) -> List[Tuple[str, bool]]:
    tfidf_guesses = model.guess([q["text"] for q in questions], BUZZ_NUM_GUESSES)
    tfidf_results = []
    for guesses in tfidf_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        tfidf_results.append((guesses[0][0], buzz))
        
    drqa_results = DrqaPredictor.batch_predict(predictor, questions)
        
#    print("length of tfidf and drqa predictions:", len(tfidf_results), len(drqa_results))
    if len(tfidf_results) != len(drqa_results):
        raise ValueError("length of predictions from two methods are different")
    outputs = []
    for i in range(len(tfidf_results)):
        if drqa_results[i][1] > DRQA_THRESHOLD:
            print("use drqa")
            outputs.append((drqa_results[i][0].replace(" ","_"), True))
        else:
            outputs.append(tfidf_results[i])
    
    return outputs

class DrqaPredictor:
    def __init__(self):
        pass
    
    def train(self):
        pass
    
    @classmethod
    def predict(cls, predictor, question_text, contexts):
        examples = []
        for context in contexts:
            examples.append((context, question_text))
#        print("now drqa-----------------")
#        print("examples", len(examples))
#        print(examples[0])
        predictions = predictor.predict_batch(
            examples, top_n=1
        )
#        print("pred", predictions)
        best_guess = None
        best_score = -1
        for j in range(len(predictions)): #for each question
    #        result = [(p[0], float(p[1])) for p in predictions[j]] #multiple guesses
            if predictions[j][0][1] > best_score:
                best_guess = predictions[j][0][0]
                best_score = predictions[j][0][1]
#        print("result:",best_guess, best_score)
        return best_guess, best_score
    
    @classmethod
    def batch_predict(cls, predictor, questions):
        examples = []
        qids = []  #number of parallel contexts of each question
        for ques in questions:
            question_text = ques["text"]
#            print(question_text)
            qids.append(len(ques["contexts"]))
            for context in ques["contexts"]:
                examples.append((context, question_text))
        predictions = predictor.predict_batch(
            examples, top_n=1
        )
#        print(qids)
        
        results = []    #a list of output results (text, score)
        t = 0
        for i in range(len(questions)):
            best_guess = None
            best_score = -1
            for j in range(qids[i]): #for each question
                if predictions[t][0][1] > best_score:
                    best_guess = predictions[t][0][0]
                    best_score = predictions[t][0][1]
                t += 1
            results.append((best_guess, best_score))
#        print(t, len(predictions))
        return results
    
    @classmethod
    def load(cls):
        drqa_predictor = Predictor(
            model=DRQA_MODEL_PATH,
            tokenizer='regexp',
            embedding_file=None,
            num_workers=0
        )
        return drqa_predictor

class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app(enable_batch=True):
    tfidf_guesser = TfidfGuesser.load()
    drqa_predictor = DrqaPredictor.load()
    print("loaded models")
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        contexts = request.json['contexts']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
        
        dq_guess, dq_score = DrqaPredictor.predict(drqa_predictor, question, contexts, BUZZ_NUM_GUESSES)
        if dq_score > DRQA_THRESHOLD:
            print("use drqa")
            return jsonify({'guess': dq_guess.replace(" ","_"), 'buzz': True, 'drqa': True})
        
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 2,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, drqa_predictor, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    print("run app")
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()

@cli.command()
def drqa_train():
    """
    Train the drqa model, requires preprocessed data
    """
    pass

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()