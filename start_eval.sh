export PYTHONPATH=`pwd`
MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
DATA_FOLDER=$2
python training_ptr_gen/eval.py -m $MODEL_PATH -d $DATA_FOLDER # > logs/eval_log.$MODEL_NAME 2>&1 &
