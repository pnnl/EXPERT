import os
import re
import sys
import json
import nltk
import torch
import numpy as np
import pandas as pd

from itertools import compress
from collections import defaultdict
from scipy.special import softmax
from pandas.io.json import json_normalize
from allennlp.predictors.predictor import Predictor
import spacy

srl_pattern = r"\[\S+:\s.+?\]"


def sentence_segmentation(doc):
    nlp = spacy.load("en_core_web_sm")
    segments = nlp(doc)
    sentences = []
    for sent in segments.sents:
        sentences.append(sent.text)
    return sentences


def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if lst_cols is not None and len(lst_cols) > 0 and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series)):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def save_json(df, fn):
    data = df.to_dict('records')
    with open(fn, 'w') as fp:
        for d in data:
            fp.write(json.dumps(d))
            fp.write("\n")


def load_json(fn):
    data = []
    with open(fn, "r") as fp:
        for lines in fp:
            data.append(json.loads(lines))
    return pd.DataFrame(data)


def run_allennlp_model(model_predictor, data):
    if len(data) != 0:
        with torch.no_grad():
            model_predictions = model_predictor.predict_batch_json(data)
        # model_predictions.to('cpu')
        torch.cuda.empty_cache()
    else:
        model_predictions = []
    return model_predictions


def run_spacy_model(spacy_pipe, data, disable):
    model_predictions = []
    for doc in spacy_pipe.pipe(data, disable=disable):
        print([(ent.text, ent.label_) for ent in doc.ents])
    return model_predictions


class Pipeline:

    def __init__(self, data_fn, save_dir="./", verbose=True, debug_mode=False, model_dir=None, overwrite_dir=False,
                 enable_save=True, use_gpu=None, text_col='report', id_col='id', batch_size=None, model='allennlp'):
        """

        :param data_fn: Input data for pipeline. Can be a pandas DataFrame, string, CSV file path, or JSON file path.
        :param save_dir: Directory to save output.
        :param verbose: Verbose flag.
        :param debug_mode: Samples first 5 instances for quick run and verification of outputs.
        :param model_dir: If not None, will use AllenNLP models from this directory. If None, will download models.
        :param overwrite_dir:
        :param enable_save: If True, will save intermediary files from each function call and final output.
                            Otherwise, nothing is written to file.
        :param use_gpu: Enable gpu usage for allenNLP models.
        :param text_col: Name of the column containing data to process.
        :param id_col: Name of the id column. Should be a unique identifier for each record.
        :param batch_size: If None, will process all documents passed.
                            If int, will process documents in batches of size batch_size.
        :param model:

        """

        # TODO: Test on small sentences (1-5 words).

        self._text_col = text_col
        self._id_col = id_col
        self._batch = batch_size
        self._over = overwrite_dir
        self._model = model
        self._allenNames = ['allennlp', "Allennlp", "AllenNLP", 'allen', "AllenNlp"]
        self._debug_value = 5

        if isinstance(data_fn, pd.DataFrame):
            self.raw_data = data_fn
            if debug_mode:
                self.raw_data = self.raw_data.iloc[:self._debug_value]
        elif ".csv" in data_fn:
            if debug_mode:
                self.raw_data = pd.read_csv(data_fn, nrows=self._debug_value)
            else:
                self.raw_data = pd.read_csv(data_fn)
            if "Unnamed: 0" in self.raw_data.columns:
                self.raw_data = self.raw_data.drop(columns=["Unnamed: 0"])
        elif ".json" in data_fn:
            try:
                self.raw_data = load_json(data_fn)
            except ValueError:
                self.raw_data = pd.read_json(data_fn, lines=True)
            if "Unnamed: 0" in self.raw_data.columns:
                self.raw_data = self.raw_data.drop(columns=["Unnamed: 0"])
            if debug_mode:
                self.raw_data = self.raw_data.iloc[:self._debug_value]
        else:
            self.raw_data = pd.DataFrame({self._text_col: data_fn}, index=[0])
            self.raw_data = self.raw_data.reset_index().rename(columns={"index": self._id_col})
            if debug_mode:
                self.raw_data = self.raw_data.iloc[:self._debug_value]

        if self._model in self._allenNames:
            self.model_dir = model_dir
            # Paths to various AllenNLP models
            if self.model_dir is None:
                self._ner_model_path = "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
                self._srl_model_path = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"
#                self._srl_model_path = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz"
                self._coref_model_path = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
                self._svo_model_path = "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
                self._sentiment_model_path = "https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
            else:
                self._ner_model_path = self.model_dir + "ner-model-2020.02.10.tar.gz"
                self._srl_model_path = self.model_dir + "bert-base-srl-2020.03.24.tar.gz"
                self._coref_model_path = self.model_dir + "coref-spanbert-large-2020.02.27.tar.gz"
                self._svo_model_path = self.model_dir + "biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
                self._sentiment_model_path = self.model_dir + "basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
        else:
            self.spacy_nlp_pipe = spacy.load("en_core_web_sm")

        self._data_fn = data_fn
        self._save_dir = save_dir
        self._verbose = verbose
        self._debug = debug_mode
        self._enable_save = enable_save
        self.order = None
        if use_gpu is None:
            self._use_gpu = -1
        else:
            self._use_gpu = use_gpu
        self.need_to_process = True

        self._data_for_predicting = None
        self._data_for_coref = None

        self.ner_df = None
        self._per_ner = None

        self._srl_df = None
        self._per_srl = None

        self._coref_df = None
        self._per_coref = None
        self.pos_df = None
        self.np_df = None

        self._svo_df = None
        self._per_svo = None

        self._sentiment_df = None
        self._per_sent = None

        self.result_df = None
        self.source_dfs = None

        if self._enable_save and self._save_dir == "":
            sys.exit("Must provide save_dir if enable_save is True")

        if self._enable_save:
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)

    def create_new_ids(self):
        """
        Create sentence level ids from the id_col given.
        Sentence ids are constructed from the document id and the sentence number in the document.
        Ex: 0_1 is the id for sentence 1 in document 0.

        :return: A new DataFrame with sentene_id column
        """
        new_col = []
        for _, row in self.raw_data.groupby("doc_id", sort=False).size().reset_index().iterrows():
            counts = row[0]
            doc_num = row["doc_id"]
            for i in range(counts):
                new_id = str(doc_num) + "_" + str(i)
                new_col.append(new_id)
        self.raw_data["sentence_id"] = new_col

    def remove_single_word_sentences_and_limit_tokens(self):
        """
        Removing single word sentences that break allennlp models.
        And limits the number of tokens in a sentence (also breaks allennlp bert-based models).
            Error Given: IndexError: index out of range in self
        Default limit is 350, but in practice 200 works better.
        Reference: https://github.com/allenai/allennlp/issues/3235
        :return: A new DataFrame with single word sentences removed.
        """

        def strip_whitespace(lst):
            not_space = []
            for l in lst:
                if (not l.isspace()) and (l != "") and (re.search('[a-zA-Z]', l)):  # A sentence must contain at least 1
                    not_space.append(l.strip())
            return not_space

        self.raw_data["temp_sent_len"] = self.raw_data["sentence"].apply(lambda x:
                                                                         len(strip_whitespace(x.split(" "))))
        self.raw_data = self.raw_data.loc[self.raw_data["temp_sent_len"] > 2].reset_index(drop=True)
        self.raw_data = self.raw_data.loc[self.raw_data["temp_sent_len"] < 350].reset_index(drop=True)
        if self._verbose and self._debug:
            print(self.raw_data['temp_sent_len'].max(), self.raw_data['temp_sent_len'].min(),
                  self.raw_data['temp_sent_len'].mean())
        self.raw_data = self.raw_data.drop(columns=["temp_sent_len"])

    def format_for_prediction(self):
        """
        Formats the input data into appropriate representation for allenNLP model processing.
        If enable_save: saves reformatted data to files. Else, sets internal variables to reformatted DataFrame.
        :return: None
        """
        self.raw_data["sentence"] = self.raw_data.apply(lambda x: sentence_segmentation(x[self._text_col]), axis=1)
        self.raw_data = explode(self.raw_data, ["sentence"])
        self.remove_single_word_sentences_and_limit_tokens()
        if len(self.raw_data) == 0:
            if self._verbose:
                print("No sentences remaining after removing 1- and 2-word sentences.")
            return
        self.raw_data["doc_id"] = self.raw_data[self._id_col]
        self.raw_data["doc_id"] = self.raw_data["doc_id"].astype(str)
        self.create_new_ids()

        # Add full document to back each row
        docs = []
        for i, grp in self.raw_data.groupby("doc_id", sort=False):
            document = " ".join(grp["sentence"].tolist())
            docs.extend([document] * len(grp))
        doc_content = pd.DataFrame(docs, columns=["document"])
        self.raw_data["document"] = docs

        # Save all formatted data

        # For NER, POS, SRL, and SVO, save only sentences to json
        content = self.raw_data[["sentence"]].to_dict("records")
        # For coref, save only documents to json
        doc_content = doc_content[["document"]].to_dict("records")

        if self._enable_save:
            self.raw_data.to_csv(self._save_dir + "reformatted_data.csv")
            save_file = self._save_dir + "data_for_predicting.json"
            with open(save_file, "w") as fp:
                for c in content:
                    fp.write(json.dumps(c))
                    fp.write("\n")

            save_file = self._save_dir + "data_for_coref_predicting.json"
            with open(save_file, "w") as fp:
                for c in doc_content:
                    fp.write(json.dumps(c))
                    fp.write("\n")
        else:
            self._data_for_predicting = []
            for c in content:
                self._data_for_predicting.append(c)
            self._data_for_coref = []
            for c in doc_content:
                self._data_for_coref.append(c)

        # Set flag to false
        self.need_to_process = False

    def describe(self):
        if self.need_to_process:
            self.format_for_prediction()

        print("Shape of self.raw_data",self.raw_data.shape)
        print("Number of sentences per document")
        print(self.raw_data.groupby('doc_id').size())
        print()
        counts = self.raw_data.groupby('doc_id').size().reset_index(name='counts')
        print("Average number of sentences", counts['counts'].mean())
        print("Max number of sentences")
        print(counts.iloc[counts['counts'].idxmax()])
        print("Min number of sentences")
        print(counts.iloc[counts['counts'].idxmin()])

        temp = self.raw_data.copy()
        temp['len'] = temp['sentence'].apply(lambda x: len(x.split(" ")))
        print("Average sentence length")
        print(temp['len'].mean())
        print("Longest sentence")
        print(temp.iloc[temp['len'].idxmax()][['doc_id', 'sentence_id', 'len']])


    def run_nlp_discovery(self, order=None):
        """
        User-level function to call different NLP functions on data.
        :param order: The order in which to process data. If None, calls all functions.
        :return: None
        """
        if order is None:
            order = ["ner", "pos", "svo", "srl", "coref", "sentiment", "np"]
        self.order = order

        not_allen_models = ['pos', 'np']

        if self._enable_save and not self._over:
            reformat_path = self._save_dir + "reformatted_data.csv"
            data_path = self._save_dir + "data_for_predicting.json"
            if os.path.exists(reformat_path) and os.path.exists(data_path):
                self.need_to_process = False
        if self.need_to_process:
            self.format_for_prediction()

        if len(self.raw_data) > 0:
            for operation in self.order:
                if operation not in not_allen_models:
                    to_continue = self.function_call(operation)
                    if not to_continue:
                        print("Failed {0}. Continuing to next function.".format(operation))
                else:
                    to_continue = getattr(self, operation)()
                    if not to_continue:
                        print("Failed {0}. Continuing to next function.".format(operation))
        else:
            if self._verbose:
                print("Not enough data. Cannot run NLP models: ", " ".join(o for o in order))

    def ner(self):
        # Function for backwards compatibility
        self.run_nlp_discovery(order=['ner'])

    def coref(self):
        # Function for backwards compatibility
        self.run_nlp_discovery(order=['coref'])

    def svo(self):
        # Function for backwards compatibility
        self.run_nlp_discovery(order=['svo'])

    def srl(self):
        # Function for backwards compatibility
        self.run_nlp_discovery(order=['srl'])

    def sentiment(self):
        # Function for backwards compatibility
        self.run_nlp_discovery(order=['sentiment'])

    def function_call(self, funct, data_file="data_for_predicting.json",
                      coref_data_file="data_for_coref_predicting.json"):
        """

        :param coref_data_file:
        :param funct:
        :param data_file:
        :return:
        """
        data_path = self._save_dir + data_file
        coref_data_path = self._save_dir + coref_data_file
        reformat_path = self._save_dir + "reformatted_data.csv"

        if self._verbose:
            print("Running AllenNLP " + funct)

        if self._enable_save:
            reformat_data = pd.read_csv(reformat_path)
        else:
            reformat_data = self.raw_data
        if "Unnamed: 0" in reformat_data.columns:
            reformat_data = reformat_data.drop(columns=["Unnamed: 0"])

        pred_ids = reformat_data["sentence_id"].tolist()
        if self._enable_save:
            data = []
            if funct == "coref":
                with open(coref_data_path, "r") as fp:
                    for lines in fp:
                        data.append(json.loads(lines))
            else:
                with open(data_path, "r") as fp:
                    for lines in fp:
                        data.append(json.loads(lines))
        else:
            if funct == "coref":
                data = self._data_for_coref
            else:
                data = self._data_for_predicting

        data = pd.DataFrame(data)

        if len(data) > 0:
            if funct == "ner":
                disable_lst = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
                predictor = Predictor.from_path(self._ner_model_path)
            elif funct == "srl":
                predictor = Predictor.from_path(self._srl_model_path)
            elif funct == 'coref':
                predictor = Predictor.from_path(self._coref_model_path)
            elif funct == 'svo':
                predictor = Predictor.from_path(self._svo_model_path)
            else:  # sentiment
                predictor = Predictor.from_path(self._sentiment_model_path)

            prediction_path = self._save_dir + funct + "_predictions.txt"
            processed_path = self._save_dir + funct + "_processed.json"

            if self._use_gpu != -1:
                cuda_device = list(range(torch.cuda.device_count()))
                if self._verbose:
                    print("Cuda devices:", cuda_device)
                    print("Chosen device:", self._use_gpu)
                torch.cuda.set_device(self._use_gpu)
                # predictor._model = predictor._model.cuda(cuda_device[self._use_gpu])
                predictor._model = predictor._model.cuda()

            if self._batch is not None:
                if self._verbose:
                    print("With batch size: {0}. Total sentences/documents: {1}".format(self._batch, len(data)))
                predictions = []
                length_of_data = len(data)
                total_batches = int(length_of_data // self._batch) + 1
                rows, counter = 0, 0
                while rows < length_of_data:
                    if self._verbose:
                        print("Starting batch {0} of {1} with batch size of {2}".format(counter,
                                                                                        total_batches - 1, self._batch))
                    temp = data.iloc[rows:rows + self._batch]
                    temp_ids = pred_ids[rows:rows + self._batch]
                    if self._model in self._allenNames:
                        preds = run_allennlp_model(predictor, temp.to_dict(orient='records'))
                    else:
                        preds = run_spacy_model(self.spacy_nlp_pipe, temp.to_dict(orient='records'), disable_lst)

                    if len(preds) != 0:
                        for p, pred in enumerate(preds):
                            pred.update({'sentence_id': temp_ids[p]})
                        predictions.append(pd.DataFrame(preds))
                    else:
                        if self._verbose:
                            print("No predictions produced.")
                    if self._verbose:
                        print("Finished Batch {0} of {1} with batch size of {2}".format(counter,
                                                                                        total_batches - 1, self._batch))
                    counter += 1
                    rows += self._batch
            else:
                predictions = run_allennlp_model(predictor, data.to_dict(orient='records'))

            if self._verbose:
                print("Predictions done!")
                print("Formatting results...")

            preds = pd.concat(predictions)

            if self._enable_save:
                preds.to_json(prediction_path, orient='records', lines=True)
                if self._verbose:
                    if self._enable_save:
                        print("Saved raw predictions to:", prediction_path)
            else:
                if funct == "ner":
                    self._per_ner = preds
                elif funct == 'srl':
                    self._per_srl = preds
                elif funct == 'coref':
                    self._per_coref = preds
                elif funct == 'svo':
                    self._per_svo = preds
                elif funct == 'sentiment':
                    self._per_sent = preds

            del preds
            del predictions

            if funct == "ner":
                if self.order != ['ner']:
                    combine_tags = False
                else:
                    combine_tags = True
                self.ner_df = self.process_ner_prediction_file(reformat_path, prediction_path,
                                                               combine_tags=combine_tags)
                self.ner_df = self.ner_df.reset_index(drop=True)
                if self._enable_save:
                    save_json(self.ner_df, processed_path)
            elif funct == 'srl':
                self._srl_df = self.process_srl_prediction_file(reformat_path, prediction_path)
                self._srl_df = self._srl_df.reset_index(drop=True)
                if self._enable_save:
                    save_json(self._srl_df, processed_path)
            elif funct == 'coref':
                self._coref_df = self.process_coref_svo_prediction_files(prediction_path, "coref")
                self.get_coref_clusters()
                if self._enable_save:
                    save_json(self._coref_df, processed_path)
            elif funct == 'svo':
                self._svo_df = self.process_coref_svo_prediction_files(prediction_path, "svo")
                self._svo_df = self._svo_df.rename(columns={"words": "token"})
                self._svo_df = self._svo_df[
                    ["pos", "predicted_dependencies", "predicted_heads", "token", "sentence_id"]]
                self._svo_df = explode(self._svo_df, ["pos", "predicted_dependencies", "predicted_heads", "token"],
                                       fill_value='')

                constructed = self.construct_svo_tuple()
                if self._enable_save:
                    save_json(self._svo_df, processed_path)
                    save_json(constructed, self._save_dir + "svo_tuples.json")
            elif funct == 'sentiment':
                self._sentiment_df = self.process_sentiment_prediction_file(reformat_path, prediction_path)
                if self._enable_save:
                    save_json(self._sentiment_df, processed_path)
            if self._verbose:
                if self._enable_save:
                    print("Saved formatted predictions to:", processed_path)
                print("Done!")
            return True
        else:
            print("Exiting. No Data.")
            return False

    def process_ner_prediction_file(self, data_file, prediction_file, combine_tags=True):

        if self._enable_save:
            data = pd.read_csv(data_file)
            predictions = []
            with open(prediction_file, "r") as fp:
                for line in fp:
                    predictions.append(pd.json_normalize(json.loads(line)))
            pred_df = pd.concat(predictions)
        else:
            data = self.raw_data
            pred_df = self._per_ner
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        pred_df.reset_index(drop=True, inplace=True)
        tags, correct = [], []
        sentence_ids, sentences = [], []
        words, confidences = [], []
        begin_idx, end_idx = [], []
        # for idx, (i, row) in zip(sent_ids, pred_df.iterrows()):
        for i, row in pred_df.iterrows():
            print(row)
            idx = row["sentence_id"]
            logits = row["logits"]
            if combine_tags:
                row_conf = []
                for w_logit in logits:
                    w_confidence = softmax(w_logit)
                    row_conf.append(np.max(w_confidence))

                # Combine tags that need combining
                joined_tags, b_tag = [], []
                begin_tag = False
                for tag_index, t in enumerate(row["tags"]):
                    if "B-" in t:
                        if begin_tag:
                            joined_tags.append(b_tag)
                        begin_tag = True
                        b_tag = [tag_index]
                    elif "I-" in t:
                        if begin_tag:
                            b_ext = row["tags"][b_tag[0]].split("-")[-1]
                            i_ext = t.split("-")[-1]
                            if b_ext == i_ext:
                                b_tag.append(tag_index)
                    elif "L-" in t:
                        if begin_tag:
                            b_ext = row["tags"][b_tag[0]].split("-")[-1]
                            l_ext = t.split("-")[-1]
                            if b_ext == l_ext:
                                b_tag.append(tag_index)
                                joined_tags.append(b_tag)
                                begin_tag = False
                            elif len(b_tag) > 1:
                                joined_tags.append(b_tag)
                                joined_tags.append([tag_index])
                                begin_tag = False
                            else:
                                joined_tags.append([tag_index])
                        else:
                            joined_tags.append([tag_index])
                    else:
                        if begin_tag:
                            joined_tags.append(b_tag)
                            begin_tag = False
                        joined_tags.append([tag_index])
                combined_all_tags = []
                combined_all_tokens = []
                avg_logits = []
                for joined in joined_tags:
                    combined_token = []
                    combined_tag = ""
                    combined_softmax = []
                    if len(joined) > 1:
                        for c in joined:
                            combined_token.append(row["words"][c])
                            combined_tag = row["tags"][c].split("-")[-1]
                            combined_softmax.append(row_conf[c])
                        combined_softmax = np.mean(np.array(combined_softmax))
                        combined_token = " ".join(combined_token)
                    else:
                        combined_token = row["words"][joined[0]]
                        combined_tag = row["tags"][joined[0]].split("-")[-1]
                        combined_softmax = row_conf[joined[0]]
                    combined_all_tokens.append(combined_token)
                    combined_all_tags.append(combined_tag)
                    avg_logits.append(combined_softmax)
                tags.extend(combined_all_tags)
                confidences.extend(avg_logits)
                sentence_ids.extend([idx] * len(combined_all_tags))
                words.extend(combined_all_tokens)

                start = 0
                for word in combined_all_tokens:
                    end = start + len(word) - 1
                    begin_idx.append(start)
                    end_idx.append(end)
                    start = end + 2
            else:
                tags.extend(row["tags"])
                sentence_ids.extend([idx] * len(row["tags"]))
                words.extend(row["words"])
                logits = list(compress(logits, row['mask']))
                for w_logit in logits:
                    w_confidence = softmax(w_logit)
                    confidences.append(np.max(w_confidence))

                start = 0
                for word in row["words"]:
                    end = start + len(word) - 1
                    begin_idx.append(start)
                    end_idx.append(end)
                    start = end + 2
        result_df = pd.DataFrame({"predicted_role": tags, "sentence_id": sentence_ids,
                                  "confidence": confidences, "token": words,
                                  "start_idx": begin_idx, "end_idx": end_idx})
        # result_df = pd.merge(result_df, data, left_on="sentence_id", right_on="id")
        result_df = pd.merge(result_df, data, on="sentence_id")
        return result_df

    def process_srl_prediction_file(self, data_file, prediction_file):
        if self._enable_save:
            data = pd.read_csv(data_file)
            predictions = []
            with open(prediction_file, "r") as fp:
                for line in fp:
                    predictions.append(json_normalize(json.loads(line)))
            pred_df = pd.concat(predictions)
        else:
            data = self.raw_data
            pred_df = self._per_srl

        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        pred_df.reset_index(drop=True, inplace=True)
        verbs = []
        for i, row in pred_df.iterrows():
            idx = row["sentence_id"]
            verb_list = row["verbs"]
            args, words = {}, []
            for j, v in enumerate(verb_list):
                tag_list = v["tags"]
                args[j] = tag_list

            if len(args) > 0:
                f = [{} for _ in range(len(args[0]))]
                for k in range(len(args)):
                    for j in range(len(args[k])):
                        f[j][k] = args[k][j]
            else:
                f = [{} for _ in range(len(row["words"]))]
            word = row["words"]
            words.extend(word)
            verbs.append(pd.DataFrame({"srl_args": f, "sentence_id": [idx] * len(f), "token": words}))
        if len(verbs) != 0:
            verb_df = pd.concat(verbs)
            verb_df = pd.merge(verb_df, data, on="sentence_id")
        else:
            data['srl_args'] = "None"
            verb_df = data
        return verb_df

    def resolve_srl_labels(self):
        tokens, roles, ids = [], [], []
        for i, row in self._srl_df.iterrows():
            idx = row["sentence_id"]
            args = row["srl_arg"]
            for a in args:
                partial = a[1:-1]
                role = partial.split(":")[0]
                token = [x for x in partial.split(":")[-1].split(" ") if x != ""]
                tokens.extend(token)
                roles.extend([role] * len(token))
                ids.extend([idx] * len(token))
        self._srl_df = pd.DataFrame({"token": tokens, "role": roles, "sentence_id": ids})

    def get_coref_clusters(self):
        # Get the same tokenizer used in the coref model to map clusters and words in doc
        tok_model = spacy.load("en_core_web_sm", disable=["ner"])
        cluster_df = self._coref_df.copy()
        cluster_df["doc_id"] = cluster_df["sentence_id"].apply(lambda x: x.rsplit("_", 1)[0])

        list_of_ids, list_of_clusters = [], []
        for i, grp in cluster_df.groupby("doc_id", sort=False):
            list_of_clusters.append(list(grp["clusters"].iloc[0]))
            list_of_ids.append(i)
        words_of_clusters, ids = [], []
        for i, cluster in zip(list_of_ids, list_of_clusters):
            # Just get the first instance of the document (possible to have more if multi-sentence document)
            doc = self.raw_data.loc[self.raw_data["doc_id"] == i, "document"].tolist()[0]
            document = tok_model(doc)
            token_map = dict(zip(list(range(len(document))), list(tok.text_with_ws for tok in document)))
            all_group_tokens = []
            for clust_id, group in enumerate(cluster):
                group_tokens = defaultdict(list)
                lst_of_toks, lst_of_tok_index = [], []
                for index in group:
                    tokens = ""
                    for t in range(index[0], index[1] + 1):
                        lst_of_tok_index.append(t)
                        tokens += "-" + token_map[t].strip()
                    tokens = tokens[1:]
                    lst_of_toks.append(tokens)
                    for t in range(index[0], index[1] + 1):
                        lst_of_toks.append(tokens)
                for idx in range(len(lst_of_tok_index)):
                    group_tokens[lst_of_tok_index[idx]].extend(list(set(lst_of_toks)))
                all_group_tokens.append(group_tokens)
            combined_clusters = defaultdict(list)
            for cluster_group in all_group_tokens:
                for k, v in cluster_group.items():
                    combined_clusters[k].append(v)
            intermediate_df = pd.DataFrame({"index": token_map.keys(), "token": token_map.values()})
            intermediate_df['cluster'] = intermediate_df['index'].map(combined_clusters)
            intermediate_df['doc_id'] = i
            words_of_clusters.append(intermediate_df)

        self._coref_df = pd.concat(words_of_clusters)
        self._coref_df = self._coref_df.loc[~self._coref_df["token"].str.isspace()].reset_index(drop=True)

    def pos(self, suffix="pos"):
        if self.need_to_process:
            self.format_for_prediction()
        if len(self.raw_data) > 0:
            list_of_sentences = self.raw_data["sentence"].tolist()
            sentence_ids = self.raw_data["sentence_id"].tolist()
            docids = self.raw_data["doc_id"].tolist()
            reports = self.raw_data[self._text_col].tolist()
            nlp_model = spacy.load("en_core_web_sm")

            if self._verbose:
                print("Beginning part of speech tagging")
            full_data = []

            for idx, sentence, docid, report in zip(sentence_ids, list_of_sentences, docids, reports):
                tokens, lemmas, pos, tags, deps, shapes, alphas, stops = [], [], [], [], [], [], [], []
                tenses, verbs = [], []
                parsed_sentence = nlp_model(sentence)
                future_1, future_2 = False, False
                for tok in parsed_sentence:
                    tokens.append(tok.text)
                    lemmas.append(tok.lemma_)
                    pos.append(tok.pos_)
                    tags.append(tok.tag_)
                    deps.append(tok.dep_)
                    shapes.append(tok.shape_)
                    alphas.append(tok.is_alpha)
                    stops.append(tok.is_stop)

                    tense_type, verb_type = self.get_tense(tok.morph)
#                    tense_type, verb_type = self.get_tense(tok, nlp_model)

                    # Find 'future' tense
                    # Ex. 'will rise', 'will likely rise'
                    if tense_type == "None" and verb_type == "infinitive" and future_1 is True:
                        tense_type = "future"
                        future_1 = False
                    if verb_type == "modifier":
                        future_1 = True

                    # Ex. 'to rise', 'is expected to rise'
                    if tense_type == "None" and verb_type == "infinitive" and future_2 is True:
                        tense_type = "future"
                        future_2 = False
                    if verb_type == "infinitive" and tok.tag_ == "TO":
                        future_2 = True
                    tenses.append(tense_type)
                    verbs.append(verb_type)
                full_data.append(pd.DataFrame({self._id_col: docid, self._text_col: report, "doc_id": docid,
                                               "sentence_id": [idx] * len(tokens), "token": tokens, "lemma": lemmas,
                                               "POS": pos, "tag": tags, "dep": deps, "shape": shapes, "alpha": alphas,
                                               "stop": stops, "tense": tenses, "verb form": verbs}))
            self.pos_df = pd.concat(full_data)
            self.pos_df = self.pos_df.loc[self.pos_df["POS"] != "SPACE"]
            self.pos_df = self.pos_df.reset_index(drop=True)
            if self._enable_save:
                save_json(self.pos_df, self._save_dir + suffix + "_processed.json")

                # Summary of tags
                new_df = pd.DataFrame({"Counts of tense types": [self.pos_df['tense'].value_counts().to_dict()],
                                       "Counts of verb forms": [self.pos_df['verb form'].value_counts().to_dict()],
                                       "Counts of tag type": [self.pos_df['tag'].value_counts().to_dict()],
                                       "Counts of stop words": [self.pos_df['stop'].apply(
                                           lambda x: "stop word" if x else "not stop word").value_counts().to_dict()],
                                       "Total number of verbs": len(
                                           self.pos_df.loc[self.pos_df['tag'].str.startswith('VB')])},
                                      index=[0])
                new_df.to_json(self._save_dir + "summary.json")

            if self._verbose:
                print("Done!")
            return True
        else:
            return False

    @staticmethod
    def get_tense(morph):
        morphology = morph.to_dict()

        tense_type='None'
        if 'Tense' in morphology.keys():
            if morphology['Tense'] == 'Pres':
                tense_type = 'present'
            elif morphology['Tense'] == 'Past':
                tense_type = 'past'


        verb_form = 'None'
        if 'VerbForm' in morphology.keys():
            if morphology['VerbForm'] == 'Inf':
                verb_form = 'infinitive'
            elif morphology['VerbForm'] == 'Part':
                verb_form = 'participle'
            elif morphology['VerbForm'] == 'Fin':
                verb_form = 'finite'
            elif morphology['VerbForm'] == 'Mod':
                verb_form = 'modifier'

        return tense_type, verb_form

        
    @staticmethod
    def get_tense_old(token, nlp_model):
        morphology = nlp_model.vocab.morphology.tag_map[token.tag_]
        if "Tense_past" in morphology.keys():
            tense_type = "past"
        elif "Tense_pres" in morphology.keys():
            tense_type = "present"
        else:
            tense_type = "None"

        if "VerbForm_inf" in morphology.keys():
            verb_form = "infinitive"
        elif "VerbForm_part" in morphology.keys():
            verb_form = "participle"
        elif "VerbForm_fin" in morphology.keys():
            verb_form = "finite"
        elif "VerbType_mod" in morphology.keys():
            verb_form = "modifier"
        else:
            verb_form = "None"
        return tense_type, verb_form

    def tense(self):
        self.pos(suffix="tense")

    def np(self):
        def add_noun_phrases(sentence):
            """
            Finds all noun phrases with optional determiner, any number of adjectives,
                and at least one noun or proper noun
            Doesn't include any prepositional phrases or subordinate clauses that modify a nominal
            Noun phrases cannot contain other noun phrases

            Args:
                sentence: str. Text to extract noun phrases from

            Returns: List. Noun phrases

            """
            #
            grammar = "NP: {<DT>?<JJ>*<CC>*<JJ>*(<NN>|<NNP>|<NNS>|<NNPS>)+(<HYPH>(<NN>|<NNP>|<NNS>|<NNPS>)+)*}"

            cp = nltk.RegexpParser(grammar)
            result = cp.parse(sentence)
            phrases = []
            for leaf in result:
                if not isinstance(leaf, tuple):
                    if leaf.label() == "NP":
                        string = " ".join([l[0][0] for l in leaf.pos()])
                        phrases.append(string)

            return phrases

        def noun_phrases():
            """
            Add noun phrases to content
            Returns: DataFrame. Same as output but with added noun phrases.

            """
            output = self.pos_df.copy()
            output["sentence"] = output.apply(lambda x: (x['token'], x['tag']), axis=1)

            def find_matches(dataframe, string_lst):
                str_count = 0
                indexes = []
                all_matches = []
                for i, row in dataframe.iterrows():
                    if str_count >= len(string_lst):
                        all_matches.append(indexes)
                        if i == len(dataframe) - 1:
                            return all_matches
                        else:
                            indexes = []
                            str_count = 0
                    if row['token'] == string_lst[str_count]:
                        indexes.append(i)
                        str_count += 1
                    else:
                        str_count = 0
                        indexes = []
                return all_matches

            new_output = []
            for j, gp in output.groupby(['sentence_id'], sort=False):
                gp = gp.reset_index(drop=True)
                sentence = gp["sentence"].tolist()
                phrase = add_noun_phrases(sentence)
                gp['noun phrases'] = ""
                for p in phrase:
                    matches = find_matches(gp, p.split(" "))
                    for m in matches:
                        gp.at[m, "noun phrases"] = p
                new_output.append(gp)
            output = pd.concat(new_output).drop(columns=["sentence"])
            return output

        if self.need_to_process:
            self.format_for_prediction()

        if self._enable_save:
            reformat_path = self._save_dir + "reformatted_data.csv"
            reformat_data = pd.read_csv(reformat_path)
        else:
            reformat_data = self.raw_data
        if "Unnamed: 0" in reformat_data.columns:
            reformat_data = reformat_data.drop(columns=["Unnamed: 0"])

        if len(reformat_data) > 0:
            if self.pos_df is None:
                self.raw_data = reformat_data
                self.pos()
            else:
                pos_fn = self._save_dir + "pos_processed.json"
                self.pos_df = load_json(pos_fn)

            if self._verbose:
                print("Beginning noun phrase extraction")
            self.np_df = noun_phrases()

            if self._enable_save:
                save_json(self.np_df, self._save_dir + "noun_phrases_processed.json")

            if self._verbose:
                print("Done!")
            return True
        else:
            return False

    def process_coref_svo_prediction_files(self, prediction_file, name):
        if self._enable_save:
            predictions = []
            with open(prediction_file, "r") as fp:
                for line in fp:
                    predictions.append(json_normalize(json.loads(line)))
            pred_df = pd.concat(predictions)
        else:
            if name == "svo":
                pred_df = self._per_svo
            else:
                pred_df = self._per_coref

        pred_df.reset_index(drop=True, inplace=True)
        return pred_df

    def construct_svo_tuple(self, svo_df=None):
        svo_tuples = []
        if svo_df is None:
            svo_df = self._svo_df

        for i, grp in svo_df.groupby('sentence_id', sort=False):
            s, v, o = None, None, None
            tuples = []
            for j, row in grp.iterrows():
                if row["predicted_dependencies"] == "root" and row["pos"] == "VERB":
                    v = row["token"]
                else:
                    if row["predicted_dependencies"] == "nsubj":
                        s = row["token"]
                    if "obj" in row["predicted_dependencies"]:
                        o = row["token"]
                if s is not None and o is not None and v is not None:
                    tuples.append((s, v, o))
                    s, v, o = None, None, None
            svo_tuples.append(pd.DataFrame({"svo": tuples, "sentence_id": [i] * len(tuples)}))
        return pd.concat(svo_tuples)

    def process_sentiment_prediction_file(self, data_file, prediction_file):
        if self._enable_save:
            data = pd.read_csv(data_file, index_col=[0])
            predictions = []
            with open(prediction_file, "r") as fp:
                for line in fp:
                    predictions.append(json_normalize(json.loads(line)))
            pred_df = pd.concat(predictions)
        else:
            data = self.raw_data
            pred_df = self._per_sent

        pred_df.reset_index(drop=True, inplace=True)
        pred_df["label"] = pred_df["label"].astype(int)
        pred_df["sentiment"] = pred_df["label"].map({1: "positive", 0: "negative"})
        pred_df = pd.merge(pred_df, data, on=['sentence_id'])
        return pred_df.drop(columns=['logits', 'token_ids', 'tokens'])

    def load_result_files(self, subset=None):
        ner_fn = self._save_dir + "ner_processed.json"
        srl_fn = self._save_dir + "srl_processed.json"
        coref_fn = self._save_dir + "coref_processed.json"
        pos_fn = self._save_dir + "pos_processed.json"
        svo_fn = self._save_dir + "svo_processed.json"
        sent_fn = self._save_dir + "sentiment_processed.json"
        np_fn = self._save_dir + "noun_phrases_processed.json"
        raw_data_fn = self._save_dir + "reformatted_data.csv"

        if subset is None:
            # Gather all results
            subset = ["ner", "srl", "pos", "svo", "coref", 'sentiment', 'np']

        try:
            self.raw_data = pd.read_csv(raw_data_fn)
            self.need_to_process = False
        except (FileNotFoundError, ValueError):
            if self._verbose:
                print("Reformatted file not found: %s. Reformatting data." % raw_data_fn)
            self.format_for_prediction()

        if "ner" in subset:
            try:
                self.ner_df = load_json(ner_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("NER file not found: %s. Running ner predictions." % ner_fn)
                self.ner()

        if "srl" in subset:
            try:
                self._srl_df = load_json(srl_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("SRL file not found: %s. Running srl predictions." % srl_fn)
                self.srl()

        if "coref" in subset:
            try:
                self._coref_df = load_json(coref_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("Coref file not found: %s. Running coref predictions." % coref_fn)
                self.coref()

        if "pos" in subset:
            try:
                self.pos_df = load_json(pos_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("POS file not found: %s. Running pos predictions." % pos_fn)
                self.pos()
        if "svo" in subset:
            try:
                self._svo_df = load_json(svo_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("SVO file not found: %s. Running svo predictions." % svo_fn)
                self.svo()

        if "sentiment" in subset:
            try:
                self._sentiment_df = load_json(sent_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("Sentiment Analysis file not found: %s. Running sentiment analysis." % sent_fn)
                self.sentiment()

        if "np" in subset:
            try:
                self.np_df = load_json(np_fn)
            except (FileNotFoundError, ValueError):
                if self._verbose:
                    print("Noun phrases file not found: %s. Running noun phrases." % np_fn)
                self.np()

    def combine_predictions_to_tuples(self, subset=None):
        if len(self.raw_data) == 0:
            if self._verbose:
                print("No predictions. Cannot combine")
            return
        if subset is None:
            subset = ['ner', 'pos', 'svo', 'srl', 'coref', 'sentiment', 'np']

        concat_lst = []
        if "ner" in subset:
            self.ner_df = self.ner_df.rename(columns={"predicted_role": "ner_predicted_role",
                                                      "confidence": "ner_confidence"})
            ner_df = self.ner_df[
                [self._id_col, self._text_col, "doc_id", "ner_predicted_role", "token",
                 "sentence_id", "ner_confidence"]].reset_index(drop=True)

            concat_lst.append(ner_df)
            drop_token = True
        else:
            # NER combines multi-token entities into one token. Need to account for this in other nlp outputs
            drop_token = False
        if "pos" in subset:
            self.pos_df = self.pos_df.rename(columns={"tag": "pos_tag"})
            if drop_token:
                pos_df = self.pos_df[[self._id_col, self._text_col, "doc_id", "sentence_id",
                                      "pos_tag", "tense", "verb form"]].reset_index(drop=True)
            else:
                pos_df = self.pos_df[[self._id_col, self._text_col, "doc_id", "sentence_id",
                                      "token", "pos_tag", "tense", "verb form"]].reset_index(drop=True)
            concat_lst.append(pos_df)
        if "svo" in subset:
            self._svo_df = self._svo_df.rename(columns={"pos": "svo_pos",
                                                        "predicted_dependencies": "svo_predicted_dependencies"})
            if drop_token:
                svo_df = self._svo_df[["sentence_id", "svo_pos", "svo_predicted_dependencies"]].reset_index(drop=True)
            else:
                svo_df = self._svo_df[["sentence_id", "token",
                                       "svo_pos", "svo_predicted_dependencies"]].reset_index(drop=True)
            concat_lst.append(svo_df)
        if "srl" in subset:
            if drop_token:
                srl_df = self._srl_df[[self._id_col, self._text_col, "doc_id",
                                       "sentence_id", "srl_args"]].reset_index(drop=True)
            else:
                srl_df = self._srl_df[[self._id_col, self._text_col, "doc_id",
                                       "sentence_id", "token", "srl_args"]].reset_index(drop=True)
            concat_lst.append(srl_df)
        if "coref" in subset:
            if drop_token:
                coref_df = self._coref_df[["doc_id", "cluster"]].reset_index(drop=True)
            else:
                coref_df = self._coref_df[["doc_id", "token", "cluster"]].reset_index(drop=True)
            concat_lst.append(coref_df)
        if "sentiment" in subset:
            sent_df = self._sentiment_df
            concat_lst.append(sent_df)
        if 'np' in subset:
            if drop_token:
                np_df = self.np_df[['doc_id', 'noun phrases']].reset_index(drop=True)
            else:
                np_df = self.np_df.reset_index(drop=True)
            concat_lst.append(np_df)

        self.result_df = pd.concat(concat_lst, axis=1)
        self.result_df = self.result_df.loc[:, ~self.result_df.columns.duplicated()]

        # Free up (gpu) memory
        del concat_lst
        del self.ner_df
        del self._srl_df
        del self._svo_df
        del self.pos_df
        del self._coref_df
        del self._sentiment_df
