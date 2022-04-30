#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import time
import argparse
from datetime import datetime

import torch
from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
from rouge import Rouge

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge
from train_util import get_input_from_batch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, data_folder, log_file_id):
        # model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (log_file_id))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        dp = config.get_data_paths(data_folder)
        self.vocab = Vocab(dp['vocab'], config.vocab_size)
        self.batcher = Batcher(dp['decode'], self.vocab, mode='decode', batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self, log_file_id):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % config.print_interval == 0:
                print('Examples %d-%d decoded in %d sec'%(counter-config.print_interval, counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single pass.")
        print("Now starting ROUGE eval...")
        rouge_1_df, rouge_2_df, rouge_l_df = self.rouge_eval(self._rouge_dec_dir, self._rouge_ref_dir)
        self.rouge_save(log_file_id, rouge_1_df, rouge_2_df, rouge_l_df)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def rouge_eval(self, decoded_dir, ref_dir):
        rouge = Rouge()
        columns=['F1','Recall','Precision']
        rouge_l_df = pd.DataFrame(columns=columns)
        rouge_1_df = pd.DataFrame(columns=columns)
        rouge_2_df = pd.DataFrame(columns=columns)

        not_found_list = []
        file_count = len(os.listdir(ref_dir))
        print('Rouge Evaluation started for {} files..'.format(file_count))
        for i in tqdm (range(file_count), desc='Running'):
            index = str(i).zfill(6)
            dec_file = decoded_dir + "/" + index + '_decoded.txt'
            ref_file = ref_dir + "/" + index + '_reference.txt'
            if os.path.isfile(dec_file) and os.path.isfile(ref_file):
                with open(dec_file, 'r') as file:
                    decoded = file.read().rstrip().decode("utf8")

                with open(ref_file, 'r') as file:
                    reference = file.read().rstrip().decode("utf8")
                
                # If somehow reference file is empty (a rare case bug, cause of which is undetected) put a placeholder.
                if reference == '':
                    reference = '[Input can not be found]' 
                score = rouge.get_scores(decoded, reference)[0]
                rouge_l_df.loc[i] = [score['rouge-l']['f'], score['rouge-l']['r'], score['rouge-l']['p']]
                rouge_1_df.loc[i] = [score['rouge-1']['f'], score['rouge-1']['r'], score['rouge-1']['p']]
                rouge_2_df.loc[i] = [score['rouge-2']['f'], score['rouge-2']['r'], score['rouge-2']['p']]
            else:
                not_found_list.append((dec_file, ref_file))
        if len(not_found_list) != 0:
            print('{} files could not be identified.'.format(len(not_found_list)))
            #print(not_found_list)
        print('Evaluation Finished..')
        return [rouge_1_df, rouge_2_df, rouge_l_df]


    def rouge_save(self, save_dir, rouge_1_df, rouge_2_df, rouge_l_df):
        save_dir = "logs/decode_"+save_dir
        if not os.path.exists(save_dir+'/rouge_scores/'):
            os.makedirs(save_dir+'/rouge_scores/')
        rouge_l_df.to_csv(save_dir+'/rouge_scores/rouge_l.csv')
        rouge_1_df.to_csv(save_dir+'/rouge_scores/rouge_1.csv')
        rouge_2_df.to_csv(save_dir+'/rouge_scores/rouge_2.csv')
        print('Rouge scores saved..')

        with open(save_dir+'/rouge_scores/summary.txt', 'w') as f:
            for df, rouge in zip([rouge_1_df, rouge_2_df,rouge_l_df], ['ROUGE-1','ROUGE-2','ROUGE-L']):
                print(rouge)
                f.write(rouge+"\n")
                for metric in rouge_l_df.columns:
                    line = "{} Mean {}".format(round(df[metric].mean(),4), metric)
                    print(line)
                    f.write(line+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decode script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument("-d",
                        dest="data_folder",
                        required=True,
                        default=None,
                        help="Dataset name 'data_T50', 'cnn' or 'movie_quotes' (default: None).")
    parser.add_argument("-l",
                        dest="log_file_id",
                        required=False,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Postfix for decode log file (default: date_time).")
    args = parser.parse_args()


    beam_Search_processor = BeamSearch(args.model_file_path, args.data_folder, args.log_file_id)
    beam_Search_processor.decode(args.log_file_id)
    
    # rouge_1_df, rouge_2_df, rouge_l_df = beam_Search_processor.rouge_eval(beam_Search_processor._rouge_dec_dir, beam_Search_processor._rouge_ref_dir)
    # beam_Search_processor.rouge_save(args.log_file_id, rouge_1_df, rouge_2_df, rouge_l_df)


