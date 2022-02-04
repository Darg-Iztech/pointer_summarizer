export PYTHONPATH=`pwd`
MODEL_PATH=$1
DATA_FOLDER=$2
python training_ptr_gen/decode.py -m $MODEL_PATH -d $DATA_FOLDER # > logs/decode_log 2>&1 &

