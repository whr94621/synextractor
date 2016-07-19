SRC_DIR=../src

python $SRC_DIR/main.py --train_embedding -f $SRC_DIR/SogouCA_seg.txt -d 50 -w 5 -t 4 &&
    --mini_count 100 
