import os
 

root_dir = os.path.expanduser("/media/darg1/Projects/pointer_summarizer/")
log_root = os.path.join(root_dir, "logs")

def get_data_paths(data_folder):
    input_dir = os.path.join(root_dir, "finished_files", data_folder)
    data_paths = {
        'chunked_train': input_dir + "/chunked/train_*",
        'train': input_dir + "/train.bin",
        'eval': input_dir + "/val.bin",
        'vocab': input_dir + "/vocab",
        'decode': input_dir + "/test.bin",
    } 
    return data_paths

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 16
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 5000 #500000
print_interval = 100
save_interval = 5000
use_gpu=True
lr_coverage=0.15
