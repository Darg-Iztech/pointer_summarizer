export PYTHONPATH=`pwd`
MODEL=$1
python training_ptr_gen/decode.py $MODEL #> logs/decode_log 2>&1 &

