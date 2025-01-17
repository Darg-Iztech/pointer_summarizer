from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch
import argparse
from datetime import datetime

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path, data_folder, log_file_id):
        dp = config.get_data_paths(data_folder)
        self.vocab = Vocab(dp['vocab'], config.vocab_size)
        self.batcher = Batcher(dp['eval'], self.vocab, mode='eval', batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (log_file_id))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = tf.summary.FileWriter(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.data[0]

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % config.print_interval == 0:
                self.summary_writer.flush()
                print('Steps {}. Elapsed time = {:.0f} seconds. Loss = {:.4f}. Avg Loss = {:.4f}.'.format(
                    iter, time.time() - start, loss, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()


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
                        help="Dataset name 'quote' or 'cnn' (default: None).")
    parser.add_argument("-l",
                    dest="log_file_id",
                    required=False,
                    default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    help="Postfix for decode log file (default: date_time).")
    args = parser.parse_args()
    eval_processor = Evaluate(args.model_file_path, args.data_folder, args.log_file_id)
    eval_processor.run_eval()


