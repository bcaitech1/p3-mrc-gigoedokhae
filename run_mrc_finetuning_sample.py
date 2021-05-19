# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ######################################################################
# 형태소분석 기반 BERT 모델 MRC Fine-tuning 샘플
# (original: Hugging-face BERT example code)
# 수정: joonho.lim
# 일자: 2019-05-27
#
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

############################################
### joonho.lim @ 2019-03-15
from korbert.src_tokenizer.tokenization_morp import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import time
import urllib3

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


class SquadExample(object):
	"""A single training/test example for the Squad dataset."""

	def __init__(self,
				 qas_id,
				 q_raw_text,
				 q_morp_token,
				 p_raw_text,
				 p_morp_token,
				 p_morp_position_list,
				 a_raw_text=None,
				 a_morp_token=None,
				 a_begin_morp=None,
				 a_end_morp=None):
		self.qas_id = qas_id
		self.q_raw_text = q_raw_text
		self.q_morp_token = q_morp_token
		self.p_raw_text = p_raw_text
		self.p_morp_token = p_morp_token
		self.p_morp_position_list = p_morp_position_list
		self.a_raw_text = a_raw_text
		self.a_morp_token = a_morp_token
		self.a_begin_morp = a_begin_morp
		self.a_end_morp = a_end_morp
		
		self.p_raw_bytes = p_raw_text.encode()
		self.p_morp_position_list.append ( len(self.p_raw_bytes) )
		
		##########################################################
		### joonho.lim @ 2019-03-15
		### check difference answer span between answer raw text and morphology anlysis boudary 
		if a_raw_text!= None and a_end_morp != None and len(p_morp_position_list) > a_end_morp :
			begin_pos = p_morp_position_list[a_begin_morp]
			end_pos = p_morp_position_list[a_end_morp+1]
			pred_answer = self.p_raw_bytes[begin_pos:end_pos].decode().strip()
			if self.a_raw_text != pred_answer :
				logger.info ( "[diff answer span] %s\t%s" % (self.a_raw_text, pred_answer) )

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		s = ""
		s += "qas_id: %s" % (self.qas_id)
		s += ", question_text: %s" % (self.q_raw_text)
		s += ", doc_text: [%s]" % (self.p_raw_text)
		if self.start_position:	s += ", start_position: %d" % (self.a_begin_morp)
		if self.start_position:	s += ", end_position: %d" % (self.a_end_morp)
		return s


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self,
				 unique_id,
				 example_index,
				 doc_span_index,
				 tokens,
				 token_to_orig_map,
				 token_is_max_context,
				 input_ids,
				 input_mask,
				 segment_ids,
				 start_position=None,
				 end_position=None):
		self.unique_id = unique_id
		self.example_index = example_index
		self.doc_span_index = doc_span_index
		self.tokens = tokens
		self.token_to_orig_map = token_to_orig_map
		self.token_is_max_context = token_is_max_context
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.start_position = start_position
		self.end_position = end_position

		
##########################################################
### joonho.lim @ 2019-03-15
### convert language anlysis result to inner data structure
def represent_ndoc ( ndoc ) :
	text = ''
	morp_list = []
	position_list = []
	
	for sentence in ndoc['sentence'] :
		text += sentence['text']
		begin_morp_id = len(morp_list)
		
		for morp in sentence['morp'] :
			morp_list.append( morp['lemma'] + '/' + morp['type'] )
			position_list.append( int(morp['position']) )
	
	return { 'text': text, 'morp_list':morp_list, 'position_list':position_list }
	
	
##########################################################
### joonho.lim @ 2019-03-15
### convert byte answer position to morphology analysis index
def mapping_answer_korquad ( p_json, rep_p, answer_text, answer_start, answer_end=-1 ) :
	if answer_end == -1 :
		answer_end = answer_start + len(answer_text)
	if answer_text != rep_p['text'][answer_start:answer_end] :
		logger.info ( '[mapping_answer_korquad error]\t%s\t%s' % (answer_text, rep_p['text'][answer_start:answer_end]) )
		return None
	
	byte_answer_start = len( rep_p['text'][:answer_start].encode() )
	byte_answer_end = len( rep_p['text'][:answer_end].encode() )
	
	base_morp_id = 0
	begin_morp_id = end_morp_id = -1
	for sent_i, sentence in enumerate( p_json['sentence'] ) :
		if byte_answer_start > sentence['morp'][-1]['position'] and byte_answer_end > sentence['morp'][-1]['position'] :
			base_morp_id += len(sentence['morp'])
			continue
			
		if begin_morp_id == -1 :
			for morp_i, morp in enumerate(sentence['morp']) :
				if morp['position'] == byte_answer_start :
					begin_morp_id = base_morp_id + morp_i
					break
				elif morp['position'] < byte_answer_start and morp_i+1 < len(sentence['morp']) and byte_answer_start < sentence['morp'][morp_i+1]['position'] :
					begin_morp_id = base_morp_id + morp_i
					logger.info ('[begin not exact match] %s\t->\t%s' % (text, morp['lemma']) )
					break
				elif morp['position'] > byte_answer_start :
					begin_morp_id = base_morp_id + morp_i
					logger.info ('[begin error] %s\t->\t%s' % (text, morp['lemma']) )
					break
		
		if begin_morp_id != -1 and end_morp_id == -1 and byte_answer_end <= sentence['morp'][-1]['position']+len(sentence['morp'][-1]['lemma'].encode()) :
			for morp_i in range( len(sentence['morp'])-1, -1, -1 ) :
				morp = sentence['morp'][morp_i]
				if morp['position'] == byte_answer_end :
					end_morp_id = base_morp_id + morp_i - 1
					break
				elif morp['position'] < byte_answer_end and byte_answer_end <= morp['position']+len(morp['lemma'].encode()) :
					end_morp_id = base_morp_id + morp_i
					break
				elif morp['position'] < byte_answer_end :
					end_morp_id = base_morp_id + morp_i
					logger.info ('[end error] %s\t->\t%s' % (text, morp['lemma']) )
					break
			
		base_morp_id += len(sentence['morp'])
	
	if begin_morp_id == -1 or end_morp_id == -1 :
		return None
	
	p_text_bytes = rep_p['text'].encode()
	begin_pos = rep_p['position_list'][begin_morp_id]
	end_pos = len(p_text_bytes)
	if end_morp_id+1 < len(rep_p['position_list']): end_pos = rep_p['position_list'][end_morp_id+1]
	pred_text = p_text_bytes[ begin_pos : end_pos ].decode().strip()
	if answer_text != pred_text :
		logger.info ('[check morp index] %s\t%s' % (text, pred_text) )
		
	return { 'begin_morp':begin_morp_id, 'end_morp':end_morp_id, 'text':answer_text }
	
	
##########################################################
### joonho.lim @ 2019-03-15
### do morphology analysis using OpenAPI service
def do_lang ( openapi_key, text ) :
	openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
	 
	requestJson = { "access_key": openapi_key, "argument": { "text": text, "analysis_code": "morp" } }
	 
	http = urllib3.PoolManager()
	response = http.request( "POST", openApiURL, headers={"Content-Type": "application/json; charset=UTF-8"}, body=json.dumps(requestJson) )

	return response.data.decode()


##########################################################
### joonho.lim @ 2019-03-15
### read squad example file and do morphology analysis
def read_squad_examples_and_do_lang(input_file, openapi_key, is_training):
	"""Read a SQuAD json file into a list of SquadExample."""
	if os.path.isfile(input_file) == False :
		raise ValueError("not exist file or folder : %s" % input_file)
		
	with open(input_file, "r", encoding='utf-8') as reader:
		input_data = json.load(reader)
	
	pqa_list = []
	for paragraphs_title in input_data['data'] :
		for pq in paragraphs_title['paragraphs'] :
			passage_text = pq['context']
			passage_lang = do_lang( openapi_key, passage_text )
			p_json = json.loads(passage_lang)['return_object']
			rep_p = represent_ndoc(p_json)
			
			for qa in pq['qas'] :
				qas_id = qa['id']
				question_text = qa['question']
				question_lang = do_lang( openapi_key, question_text )
				q_json = json.loads(question_lang)['return_object']
				rep_q = represent_ndoc(q_json)
				
				rep_a = {}
				if is_training :
					rep_a = mapping_answer_korquad( p_json, rep_p, qa['answers'][0]['text'], qa['answers'][0]['answer_start'] )
			
				pqa_list.append( {'id':qas_id, 'passage':rep_p, 'question':rep_q, 'answer':rep_a} )
	
	return read_squad_examples(pqa_list, is_training)


def read_squad_examples(input_data, is_training):
	"""Read a SQuAD json file into a list of SquadExample."""
	examples = []
	for pqa in input_data :
		a_raw_text = None
		a_morp_token = None
		a_begin_morp = None
		a_end_morp = None
		if is_training:
			a_raw_text = pqa['answer']['text']
			a_begin_morp = pqa['answer']['begin_morp']
			a_end_morp = pqa['answer']['end_morp']
			a_morp_token = pqa['passage']['morp_list'][a_begin_morp : a_end_morp+1]
			
		example = SquadExample(
			qas_id = pqa['id'],
			q_raw_text = pqa['question']['text'],
			q_morp_token = pqa['question']['morp_list'],
			p_raw_text = pqa['passage']['text'],
			p_morp_token = pqa['passage']['morp_list'],
			p_morp_position_list = pqa['passage']['position_list'],
			a_raw_text = a_raw_text,
			a_morp_token = a_morp_token,
			a_begin_morp = a_begin_morp,
			a_end_morp = a_end_morp)
		examples.append(example)
		
	logger.info( 'len(examples) : %d'% len(examples) )
	return examples		
		

def convert_examples_to_features(examples, tokenizer, max_seq_length,
								 doc_stride, max_query_length, is_training, verbose=True):
	"""Loads a data file into a list of `InputBatch`s."""

	unique_id = 1000000000

	features = []
	for (example_index, example) in enumerate(examples):
		query_tokens = []
		for q_morp in example.q_morp_token :
			query_tokens.extend( tokenizer.tokenize(q_morp) )
		if len(query_tokens) > max_query_length:
			query_tokens = query_tokens[0:max_query_length]
		
		tok_to_orig_index = []
		orig_to_tok_index = []
		all_doc_tokens = []
		for (i, token) in enumerate(example.p_morp_token):
			orig_to_tok_index.append(len(all_doc_tokens))
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				tok_to_orig_index.append(i)
				all_doc_tokens.append(sub_token)

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[ example.a_begin_morp ]
			if example.a_end_morp+1 < len(orig_to_tok_index) :
				tok_end_position = orig_to_tok_index[ example.a_end_morp+1 ] - 1
			else :
				tok_end_position = orig_to_tok_index[ -1 ]

				
		# The -3 accounts for [CLS], [SEP] and [SEP]
		max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

		# We can have documents that are longer than the maximum sequence length.
		# To deal with this we do a sliding window approach, where we take chunks
		# of the up to our max length with a stride of `doc_stride`.
		_DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
			"DocSpan", ["start", "length"])
		doc_spans = []
		start_offset = 0
		while start_offset < len(all_doc_tokens):
			length = len(all_doc_tokens) - start_offset
			if length > max_tokens_for_doc:
				length = max_tokens_for_doc
			doc_spans.append(_DocSpan(start=start_offset, length=length))
			if start_offset + length == len(all_doc_tokens):
				break
			start_offset += min(length, doc_stride)

		for (doc_span_index, doc_span) in enumerate(doc_spans):
			tokens = []
			token_to_orig_map = {}
			token_is_max_context = {}
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in query_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for i in range(doc_span.length):
				split_token_index = doc_span.start + i
				token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

				is_max_context = _check_is_max_context(doc_spans, doc_span_index,
													   split_token_index)
				token_is_max_context[len(tokens)] = is_max_context
				tokens.append(all_doc_tokens[split_token_index])
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			start_position = None
			end_position = None
			if is_training:
				# For training, if our document chunk does not contain an annotation
				# we throw it out, since there is nothing to predict.
				doc_start = doc_span.start
				doc_end = doc_span.start + doc_span.length - 1
				if (example.a_begin_morp < doc_start or
						example.a_end_morp < doc_start or
						example.a_begin_morp > doc_end or example.a_end_morp > doc_end):
					continue

				doc_offset = len(query_tokens) + 2
				start_position = tok_start_position - doc_start + doc_offset
				end_position = tok_end_position - doc_start + doc_offset

			if verbose == True and example_index < 10:
				logger.info("*** Example ***")
				logger.info("unique_id: %s" % (unique_id))
				logger.info("example_index: %s" % (example_index))
				logger.info("doc_span_index: %s" % (doc_span_index))
				logger.info("tokens: %s" % " ".join(tokens))
				logger.info("token_to_orig_map: %s" % " ".join([
					"%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
				logger.info("token_is_max_context: %s" % " ".join([
					"%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
				]))
				logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				logger.info(
					"input_mask: %s" % " ".join([str(x) for x in input_mask]))
				logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
				if is_training:
					answer_text = " ".join(tokens[start_position:(end_position + 1)])
					logger.info("start_position: %d" % (start_position))
					logger.info("end_position: %d" % (end_position))
					logger.info("answer: %s" % (answer_text))
					logger.info("orig_answer: %s" % (example.a_raw_text))

			features.append(
				InputFeatures(
					unique_id=unique_id,
					example_index=example_index,
					doc_span_index=doc_span_index,
					tokens=tokens,
					token_to_orig_map=token_to_orig_map,
					token_is_max_context=token_is_max_context,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					start_position=start_position,
					end_position=end_position))
			unique_id += 1

	return features



def _check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""

	# Because of the sliding window approach taken to scoring documents, a single
	# token can appear in multiple documents. E.g.
	#  Doc: the man went to the store and bought a gallon of milk
	#  Span A: the man went to the
	#  Span B: to the store and bought
	#  Span C: and bought a gallon of
	#  ...
	#
	# Now the word 'bought' will have two scores from spans B and C. We only
	# want to consider the score with "maximum context", which we define as
	# the *minimum* of its left and right context (the *sum* of left and
	# right context will always be the same, of course).
	#
	# In the example the maximum context for 'bought' would be span C since
	# it has 1 left context and 3 right context, while span B has 4 left context
	# and 0 right context.
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span.start + doc_span.length - 1
		if position < doc_span.start:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span.start
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index

	return cur_span_index == best_span_index



RawResult = collections.namedtuple("RawResult",
								   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
					  max_answer_length, do_lower_case, output_prediction_file,
					  output_nbest_file, verbose_logging):
	"""Write final predictions to the json file."""
	logger.info("Writing predictions to: %s" % (output_prediction_file))
	logger.info("Writing nbest to: %s" % (output_nbest_file))

	(all_predictions, all_nbest_json) = get_predictions( 
					all_examples, all_features, all_results, n_best_size,
					  max_answer_length, do_lower_case, verbose_logging )
					  
	with open(output_prediction_file, "w") as writer:
		writer.write(json.dumps(all_predictions, indent=4) + "\n")

	with open(output_nbest_file, "w") as writer:
		writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

		
def get_predictions(all_examples, all_features, all_results, n_best_size,
					  max_answer_length, do_lower_case, verbose_logging):

	example_index_to_features = collections.defaultdict(list)
	for feature in all_features:
		example_index_to_features[feature.example_index].append(feature)

	unique_id_to_result = {}
	for result in all_results:
		unique_id_to_result[result.unique_id] = result

	_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
		"PrelimPrediction",
		["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

	all_predictions = collections.OrderedDict()
	all_nbest_json = collections.OrderedDict()
	for (example_index, example) in enumerate(all_examples):
		features = example_index_to_features[example_index]

		prelim_predictions = []
		for (feature_index, feature) in enumerate(features):
			result = unique_id_to_result[feature.unique_id]

			start_indexes = _get_best_indexes(result.start_logits, n_best_size)
			end_indexes = _get_best_indexes(result.end_logits, n_best_size)
			for start_index in start_indexes:
				for end_index in end_indexes:
					# We could hypothetically create invalid predictions, e.g., predict
					# that the start of the span is in the question. We throw out all
					# invalid predictions.
					if start_index >= len(feature.tokens):
						continue
					if end_index >= len(feature.tokens):
						continue
					if start_index not in feature.token_to_orig_map:
						continue
					if end_index not in feature.token_to_orig_map:
						continue
					if not feature.token_is_max_context.get(start_index, False):
						continue
					if end_index < start_index:
						continue
					length = end_index - start_index + 1
					if length > max_answer_length:
						continue
					prelim_predictions.append(
						_PrelimPrediction(
							feature_index=feature_index,
							start_index=start_index,
							end_index=end_index,
							start_logit=result.start_logits[start_index],
							end_logit=result.end_logits[end_index]))

		prelim_predictions = sorted(
			prelim_predictions,
			key=lambda x: (x.start_logit + x.end_logit),
			reverse=True)

		_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
			"NbestPrediction", ["text", "start_logit", "end_logit"])

		seen_predictions = {}
		nbest = []
		for pred in prelim_predictions:
			if len(nbest) >= n_best_size:
				break
			feature = features[pred.feature_index]

			##########################################################
			### joonho.lim @ 2019-03-15
			### extract final text
			tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
			orig_doc_start = feature.token_to_orig_map[pred.start_index]
			orig_doc_end = feature.token_to_orig_map[pred.end_index]
			
			p_begin_morp = example.p_morp_position_list[ orig_doc_start ]
			p_end_morp = example.p_morp_position_list[ orig_doc_end+1 ]
			
			final_text = example.p_raw_bytes[ p_begin_morp:p_end_morp ].decode().strip()
			
			if final_text in seen_predictions:
				continue

			seen_predictions[final_text] = True
			nbest.append(
				_NbestPrediction(
					text=final_text,
					start_logit=pred.start_logit,
					end_logit=pred.end_logit))

		# In very rare edge cases we could have no valid predictions. So we
		# just create a nonce prediction in this case to avoid failure.
		if not nbest:
			nbest.append(
				_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

		assert len(nbest) >= 1

		total_scores = []
		for entry in nbest:
			total_scores.append(entry.start_logit + entry.end_logit)

		probs = _compute_softmax(total_scores)

		nbest_json = []
		for (i, entry) in enumerate(nbest):
			output = collections.OrderedDict()
			output["text"] = entry.text
			output["probability"] = probs[i]
			output["start_logit"] = entry.start_logit
			output["end_logit"] = entry.end_logit
			nbest_json.append(output)

		assert len(nbest_json) >= 1

		all_predictions[example.qas_id] = nbest_json[0]["text"]
		all_nbest_json[example.qas_id] = nbest_json

	return (all_predictions, all_nbest_json)

	
def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
	"""Project the tokenized prediction back to the original text."""

	# When we created the data, we kept track of the alignment between original
	# (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
	# now `orig_text` contains the span of our original text corresponding to the
	# span that we predicted.
	#
	# However, `orig_text` may contain extra characters that we don't want in
	# our prediction.
	#
	# For example, let's say:
	#   pred_text = steve smith
	#   orig_text = Steve Smith's
	#
	# We don't want to return `orig_text` because it contains the extra "'s".
	#
	# We don't want to return `pred_text` because it's already been normalized
	# (the SQuAD eval script also does punctuation stripping/lower casing but
	# our tokenizer does additional normalization like stripping accent
	# characters).
	#
	# What we really want to return is "Steve Smith".
	#
	# Therefore, we have to apply a semi-complicated alignment heruistic between
	# `pred_text` and `orig_text` to get a character-to-charcter alignment. This
	# can fail in certain cases in which case we just return `orig_text`.

	def _strip_spaces(text):
		ns_chars = []
		ns_to_s_map = collections.OrderedDict()
		for (i, c) in enumerate(text):
			if c == " ":
				continue
			ns_to_s_map[len(ns_chars)] = i
			ns_chars.append(c)
		ns_text = "".join(ns_chars)
		return (ns_text, ns_to_s_map)

	# We first tokenize `orig_text`, strip whitespace from the result
	# and `pred_text`, and check if they are the same length. If they are
	# NOT the same length, the heuristic has failed. If they are the same
	# length, we assume the characters are one-to-one aligned.
	tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

	tok_text = " ".join(tokenizer.tokenize(orig_text))

	start_position = tok_text.find(pred_text)
	if start_position == -1:
		if verbose_logging:
			logger.info(
				"Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
		return orig_text
	end_position = start_position + len(pred_text) - 1

	(orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
	(tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

	if len(orig_ns_text) != len(tok_ns_text):
		if verbose_logging:
			logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
							orig_ns_text, tok_ns_text)
		return orig_text

	# We then project the characters in `pred_text` back to `orig_text` using
	# the character-to-character alignment.
	tok_s_to_ns_map = {}
	for (i, tok_index) in tok_ns_to_s_map.items():
		tok_s_to_ns_map[tok_index] = i

	orig_start_position = None
	if start_position in tok_s_to_ns_map:
		ns_start_position = tok_s_to_ns_map[start_position]
		if ns_start_position in orig_ns_to_s_map:
			orig_start_position = orig_ns_to_s_map[ns_start_position]

	if orig_start_position is None:
		if verbose_logging:
			logger.info("Couldn't map start position")
		return orig_text

	orig_end_position = None
	if end_position in tok_s_to_ns_map:
		ns_end_position = tok_s_to_ns_map[end_position]
		if ns_end_position in orig_ns_to_s_map:
			orig_end_position = orig_ns_to_s_map[ns_end_position]

	if orig_end_position is None:
		if verbose_logging:
			logger.info("Couldn't map end position")
		return orig_text

	output_text = orig_text[orig_start_position:(orig_end_position + 1)]
	return output_text


def _get_best_indexes(logits, n_best_size):
	"""Get the n-best logits from a list."""
	index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

	best_indexes = []
	for i in range(len(index_and_score)):
		if i >= n_best_size:
			break
		best_indexes.append(index_and_score[i][0])
	return best_indexes


def _compute_softmax(scores):
	"""Compute softmax probability over raw logits."""
	if not scores:
		return []

	max_score = None
	for score in scores:
		if max_score is None or score > max_score:
			max_score = score

	exp_scores = []
	total_sum = 0.0
	for score in scores:
		x = math.exp(score - max_score)
		exp_scores.append(x)
		total_sum += x

	probs = []
	for score in exp_scores:
		probs.append(score / total_sum)
	return probs

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x


	
def parser_add_argument	( parser ) :
	## Required parameters
	parser.add_argument("--openapi_key", default="c580aac7-6772-4166-ab6d-8ecafd024d97", type=str, required=True, help="OpenAPI key information for morphology analysis")
	parser.add_argument("--bert_model", default="bert-large-cased", type=str, required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
						"bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
						"bert-base-multilingual-cased, bert-base-chinese.")
	
	## Other parameters
	parser.add_argument("--bert_model_file", default="/opt/ml/code/korbert/pytorch_model.bin", type=str, help="specific model file (i.e. pytorch_model.bin)")
	parser.add_argument("--vocab_file", default="/opt/ml/code/korbert/vocab.korean_morp.list", type=str, help="specific vocab file (i.e. vocab.txt)")
	parser.add_argument("--output_dir", default="/opt/ml/outputs/models/KorBERT", type=str, 
						help="The output directory where the model checkpoints and predictions will be written.")

	parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
	parser.add_argument("--predict_file", default="/opt/ml/input/data/test_dataset", type=str,
						help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
	parser.add_argument("--eval_folder", default=None, type=str, help="SQuAD json folder for test performance")
	parser.add_argument("--max_seq_length", default=384, type=int,
						help="The maximum total input sequence length after WordPiece tokenization. Sequences "
							 "longer than this will be truncated, and sequences shorter than this will be padded.")
	parser.add_argument("--doc_stride", default=128, type=int,
						help="When splitting up a long document into chunks, how much stride to take between chunks.")
	parser.add_argument("--max_query_length", default=64, type=int,
						help="The maximum number of tokens for the question. Questions longer than this will "
							 "be truncated to this length.")
	parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
	parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
	parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
	parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs", default=3.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion", default=0.1, type=float,
						help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
							 "of training.")
	parser.add_argument("--n_best_size", default=20, type=int,
						help="The total number of n-best predictions to generate in the nbest_predictions.json "
							 "output file.")
	parser.add_argument("--max_answer_length", default=30, type=int,
						help="The maximum length of an answer that can be generated. This is needed because the start "
							 "and end predictions are not conditioned on one another.")
	parser.add_argument("--verbose_logging", action='store_true',
						help="If true, all of the warnings related to data processing will be printed. "
							 "A number of warnings are expected for a normal SQuAD evaluation.")
	parser.add_argument("--no_cuda",
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Whether to lower case the input text. True for uncased models, False for cased models.")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	return parser
	
def main():
	parser = argparse.ArgumentParser()
	parser = parser_add_argument( parser )
	args = parser.parse_args()

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
							args.gradient_accumulation_steps))

	args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_predict :
		raise ValueError("At least one of `do_train` or `do_predict` must be True.")

	if args.do_train:
		if not args.train_file:
			raise ValueError(
				"If `do_train` is True, then `train_file` must be specified.")
	if args.do_predict:
		if not args.predict_file:
			raise ValueError(
				"If `do_predict` is True, then `predict_file` must be specified.")

	if args.do_train :
		if os.path.exists(args.output_dir) and os.listdir(args.output_dir) :
			logger.error("Output directory () already exists and is not empty.")
		os.makedirs(args.output_dir, exist_ok=True)

	tokenizer_path = args.bert_model
	if args.vocab_file != None :
		tokenizer_path = os.path.join(args.bert_model, args.vocab_file)
	tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)

	train_examples = None
	num_train_steps = None
	if args.do_train:
		#########################################
		### joonho.lim @ 2019-03-15
		train_examples = read_squad_examples_and_do_lang( input_file=args.train_file, openapi_key=args.openapi_key, is_training=True )
		num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
		
		# joonho.lim - for small-size sample training
		if num_train_steps == 0 :
			num_train_steps = 1

	# Prepare model
	state_dict = None
	if args.bert_model_file != None :
		state_dict = torch.load( os.path.join(args.bert_model, args.bert_model_file) )
	model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=state_dict,
				cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

	if args.fp16:
		model.half()
	model.to(device)
	if args.local_rank != -1:
		try:
			from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())

	# hack to remove pooler, which is not used
	# thus it produce None grad that break apex
	param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	t_total = num_train_steps
	if args.local_rank != -1:
		t_total = t_total // torch.distributed.get_world_size()
	if args.fp16:
		try:
			from apex.optimizers import FP16_Optimizer
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters,
							  lr=args.learning_rate,
							  bias_correction=False,
							  max_grad_norm=1.0)
		if args.loss_scale == 0:
			optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
	else:
		optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=args.learning_rate,
							 warmup=args.warmup_proportion,
							 t_total=t_total)

	global_step = 0
	if args.do_train:
		cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
			list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
		train_features = None
		try:
			with open(cached_train_features_file, "rb") as reader:
				train_features = pickle.load(reader)
		except:
			train_features = convert_examples_to_features(
				examples=train_examples,
				tokenizer=tokenizer,
				max_seq_length=args.max_seq_length,
				doc_stride=args.doc_stride,
				max_query_length=args.max_query_length,
				is_training=True)
			if args.local_rank == -1 or torch.distributed.get_rank() == 0:
				logger.info("  Saving train features into cached file %s", cached_train_features_file)
				with open(cached_train_features_file, "wb") as writer:
					pickle.dump(train_features, writer)
		logger.info("***** Training *****")
		logger.info("  Num orig examples = %d", len(train_examples))
		logger.info("  Num split examples = %d", len(train_features))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_steps)
		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
		all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
								   all_start_positions, all_end_positions)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

	model.train()
	for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
		for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
			if n_gpu == 1:
				batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
			input_ids, input_mask, segment_ids, start_positions, end_positions = batch
			loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
			if n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu.
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				optimizer.backward(loss)
			else:
				loss.backward()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				# modify learning rate with special warm up BERT uses
				lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr_this_step
				optimizer.step()
				optimizer.zero_grad()
				global_step += 1

	# Save a trained model
	model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
	output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
	if args.do_train:
		torch.save(model_to_save.state_dict(), output_model_file)

	# Load a trained model that you have fine-tuned
	model_state_dict = torch.load(output_model_file)
	model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=model_state_dict)
	model.to(device)
		
	if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		#########################################
		### joonho.lim @ 2019-03-15
		eval_examples = read_squad_examples_and_do_lang( input_file=args.predict_file, openapi_key=args.openapi_key, is_training=False )
		eval_features = convert_examples_to_features(
			examples=eval_examples,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			doc_stride=args.doc_stride,
			max_query_length=args.max_query_length,
			is_training=False)

		logger.info("***** Running predictions *****")
		logger.info("  Num orig examples = %d", len(eval_examples))
		logger.info("  Num split examples = %d", len(eval_features))
		logger.info("  Batch size = %d", args.predict_batch_size)

		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
		all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
		# Run prediction for full data
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

		model.eval()
		all_results = []
		logger.info("Start evaluating")
		for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
			if len(all_results) % 1000 == 0:
				logger.info("Processing example: %d" % (len(all_results)))
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			with torch.no_grad():
				batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
			for i, example_index in enumerate(example_indices):
				start_logits = batch_start_logits[i].detach().cpu().tolist()
				end_logits = batch_end_logits[i].detach().cpu().tolist()
				eval_feature = eval_features[example_index.item()]
				unique_id = int(eval_feature.unique_id)
				all_results.append(RawResult(unique_id=unique_id,
											 start_logits=start_logits,
											 end_logits=end_logits))
		output_prediction_file = os.path.join(args.output_dir, "predictions.json")
		output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
		write_predictions(eval_examples, eval_features, all_results,
						  args.n_best_size, args.max_answer_length,
						  args.do_lower_case, output_prediction_file,
						  output_nbest_file, args.verbose_logging)
		
		
			
if __name__ == "__main__":
	main()
