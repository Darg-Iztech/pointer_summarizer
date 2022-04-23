export PYTHONPATH=`pwd`
MODEL_PATH=$1
DATA_FOLDER=$2
LOG_FOLDER=$3
python training_ptr_gen/train.py -m $MODEL_PATH -d $DATA_FOLDER -l $LOG_FOLDER -c # > logs/training_log 2>&1 &
