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

DRQA_MODEL_PATH = 'entity_drqa_1.mdl'
MODEL_PATH = 'tfidf2.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.2
DRQA_THRESHOLD = 0.99
possible_answers = None

def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    
    return guesses[0][0], buzz


def batch_guess_and_buzz_drqa(model, predictor, questions) -> List[Tuple[str, bool]]:
    all_paras = [q['wiki_paragraphs'] for q in questions]
    tfidf_guesses = model.guess_with_context([q["text"] for q in questions], all_paras, BUZZ_NUM_GUESSES)
    tfidf_results = []
    for guesses in tfidf_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        tfidf_results.append((guesses[0][0], buzz))
        
    drqa_results = DrqaPredictor.batch_predict(predictor, questions)
        
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

def batch_guess_and_buzz_context(tfidf_model, drqa_model, questions):
    context_guesses = tfidf_model.guess_with_context([q["text"] for q in questions], 
                                             [q['wiki_paragraphs'] for q in questions], BUZZ_NUM_GUESSES)
    question_guesses = tfidf_model.guess([q["text"] for q in questions], BUZZ_NUM_GUESSES)
    
#    drqa_guesses = None
#    if drqa_model is not None:
#        drqa_guesses = DrqaPredictor.batch_predict(drqa_model, questions)
#    print("context",context_guesses)
#    print("question",question_guesses)
#    print("drqa",drqa_guesses)
    outputs = []
    final_guesses = []
    for i in range(len(questions)):
        drqa_guess = None
        if question_guesses[i][0][1] < 0.5 and drqa_model is not None:
            print("call drqa", end=",")
            drqa_guess = DrqaPredictor.predict(drqa_model, questions[i]["text"], questions[i]["wiki_paragraphs"])
        if drqa_guess:
            final_guesses.append(combine_guesses(question_guesses[i], context_guesses[i], drqa_guess))
        else:
            final_guesses.append(combine_guesses(question_guesses[i], context_guesses[i]))
#    print("final", final_guesses)
    for guesses in final_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
#        buzz = guesses[0][1] >= 1.5
        outputs.append((guesses[0][0], buzz))

    return outputs

#for a question, combine different guesses
def combine_guesses(question_guesses, context_guesses, drqa_guesses=None):
    global possible_answers
    if drqa_guesses is not None:
        if len(context_guesses) != 0:
            guesses = [g[0] for g in question_guesses]
            scores = [g[1] for g in question_guesses]
            for guess in context_guesses:
                if guess[0] in guesses:
                    scores[guesses.index(guess[0])] += guess[1]
                else:
                    guesses.append(guess[0])
                    scores.append(guess[1])
            for guess in drqa_guesses:
                if guess[0] in possible_answers:
                    s = guess[1]
                    if guess[1] < 0.85:
                        s *= 0.2
                    if guess[0] in guesses:
                        scores[guesses.index(guess[0])] += s
                    else:
                        guesses.append(guess[0])
                        scores.append(s)
            return list(sorted(zip(guesses, scores), key=lambda x: x[1], reverse=True))
        else:
            guesses = [g[0] for g in question_guesses]
            scores = [g[1] for g in question_guesses]
            for guess in drqa_guesses:
                if guess[0] in possible_answers:
                    s = guess[1]
                    if guess[1] < 0.85:
                        s *= 0.2
                    if guess[0] in guesses:
                        scores[guesses.index(guess[0])] += s
                    else:
                        guesses.append(guess[0])
                        scores.append(s)
            return list(sorted(zip(guesses, scores), key=lambda x: x[1], reverse=True))
    
    else:
        if len(context_guesses) != 0:
            guesses = [g[0] for g in question_guesses]
            scores = [g[1] for g in question_guesses]
            for guess in context_guesses:
                if guess[0] in guesses:
                    scores[guesses.index(guess[0])] += guess[1]
                else:
                    guesses.append(guess[0])
                    scores.append(guess[1])
            
            return list(sorted(zip(guesses, scores), key=lambda x: x[1], reverse=True))
        else:
            return question_guesses
    

class DrqaPredictor:
    def __init__(self):
        pass
    
    def train(self):
        pass
    
    @classmethod
    def predict(cls, predictor, question_text, contexts):
        if len(contexts) == 0:
            return None
        
        examples = []
#        for context in contexts:
#            examples.append((context, question_text))
        for article in contexts:
            entities = ""
            for paras in article:
                for ent in paras["entities"]:
                    if ent[0] is not None:
                        e = ent[0]
                        entities += (e + ",")
            if len(entities) > 0:
                examples.append((entities, question_text))
        if len(examples) == 0:
            return None

        predictions = predictor.predict_batch(
            examples, top_n=1
        )
        
        guesses = [(pred[0][0], pred[0][1]) for pred in predictions]
        guesses.sort(key=lambda x: x[1], reverse=True)
        
        if len(guesses) > 10:
            return guesses[:10]
        else:
            return guesses
            
#        print("pred", predictions)
#        best_guess = None
#        best_score = -1
#        for j in range(len(predictions)): #for each question
#    #        result = [(p[0], float(p[1])) for p in predictions[j]] #multiple guesses
#            if predictions[j][0][1] > best_score:
#                best_guess = predictions[j][0][0]
#                best_score = predictions[j][0][1]
##        print("result:",best_guess, best_score)
#        return best_guess, best_score
    
    @classmethod
    def batch_predict(cls, predictor, questions):
        examples = []
        qids = []
        for i in range(len(questions)):
            ques = questions[i]
            question_text = ques["text"]
            contexts = ques['wiki_paragraphs']
            if len(contexts) > 0:
                for article in contexts:
                    entities = ""
                    for paras in article:
                        for ent in paras["entities"]:
                            if ent[0] is not None:
                                e = ent[0]
                                entities += (e + ",")
                    if len(entities) > 0:
                        examples.append((entities, question_text))
                        qids.append(i)
                    else:
                        examples.append((question_text, question_text))
                        qids.append(i) 
            else:
               examples.append((question_text, question_text))
               qids.append(i) 
                    
#        print("examples",examples)
#        print("qids",qids)
        batch_size = 128
        results = {}
        for i in tqdm(range(0, len(examples), batch_size)):
            predictions = predictor.predict_batch(
                examples[i:min(i+batch_size, len(examples))], top_n=2
            )
#            print("pred:",predictions)
            for j in range(len(predictions)):
                result = [(p[0], float(p[1])) for p in predictions[j]]
                qid = qids[i+j]
                if qid not in results.keys():
                    results[qid] = []
                results[qid].append(result)
#        print("res",results)
        outputs = []    #a list of output results (text, score)
        for i in range(len(questions)):
            q_res = [(r[0].replace(" ","_"), r[1]) for subres in results[i] for r in subres]
            q_res.sort(key=lambda x: x[1], reverse=True)
#            print(q_res)
            outputs.append(q_res)
#        t = 0
#        for i in range(len(questions)):
#            best_guess = None
#            best_score = -1
#            for j in range(qids[i]): #for each question
#                if predictions[t][0][1] > best_score:
#                    best_guess = predictions[t][0][0]
#                    best_score = predictions[t][0][1]
#                t += 1
#            results.append((best_guess, best_score))
#        print(t, len(predictions))
        return outputs
    
    @classmethod
    def load(cls):
        drqa_predictor = Predictor(
            model=DRQA_MODEL_PATH,
            tokenizer='regexp',
            embedding_file=None,
            num_workers=0
        )
        return drqa_predictor

class TfidfContextGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None
        self.ans_to_i = None
    
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

        self.ans_to_i = {ans: i for i, ans in enumerate(y_array)}
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
    
    def guess_with_context(self, questions: List[str], contexts, max_n_guesses: Optional[int]):
        candidates = []
#        print(contexts)
        for con in contexts:
            if len(con) == 0:
                candidates.append([])
            else:
                cand = []
                for c in con:
                    for para in c:
                        for en in para["entities"]:
                            if en[0] is not None and en[0].replace(" ", "_") not in cand:
                                cand.append(en[0].replace(" ", "_")) 
#                print(len(cand))
                candidates.append(cand)
                
        return self.guess_with_cand(questions, candidates, max_n_guesses)
    
    def guess_with_cand(self, questions: List[str], candidates: List[List[str]], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guesses = []
        for i in range(len(questions)):
#            print("candidates number:",len(candidates[i]))
#            print(candidates[i])
            cands = [cand for cand in candidates[i] if cand in self.ans_to_i.keys()]
            cand_ind = [self.ans_to_i[c] for c in cands]
            scores = [guess_matrix[i, j] for j in cand_ind]
            guess = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
            guesses.append(guess[:max_n_guesses])
#        for i in range(len(questions)):
#            guess = []
#            for cand in candidates[i]:
#                if cand is not None:
#                    cand1 = cand.replace(" ", "_")
#                    if cand1 in self.ans_to_i.keys():
#                        guess.append((cand1, guess_matrix[i, self.ans_to_i[cand1]]))
#            guess.sort(key=lambda x: x[1], reverse=True)
#            guesses.append(guess[:10])
#            if i % 500 == 0:
#                print(i, ":")
#                print(guess)
#        print(len(guesses))
        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'ans_to_i': self.ans_to_i,
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfContextGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.ans_to_i = params['ans_to_i']
            guesser.i_to_ans = params['i_to_ans']
            return guesser

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
    tfidf_guesser = TfidfContextGuesser.load()
    drqa_predictor = DrqaPredictor.load()
    global possible_answers
    possible_answers = tfidf_guesser.ans_to_i.keys()
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
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz_context(tfidf_guesser, drqa_predictor, questions)
        ])
    
#    @app.route('/api/1.0/quizbowl/batch_act_cand', methods=['POST'])
#    def batch_act_cand():
#        questions = [q for q in request.json['questions']]
#        return jsonify([
#            {'guess': guess, 'buzz': True if buzz else False}
#            for guess, buzz in batch_guess_and_buzz_cand(tfidf_guesser, questions)
#        ])

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
def train_entities():
    """
    Train the tfidf model with context
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfContextGuesser()
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
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()