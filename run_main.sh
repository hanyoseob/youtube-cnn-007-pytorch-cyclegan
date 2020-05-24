# !bash ./run_main.sh

#python3 './main.py' \
#        --mode 'train' \
#        --lr 2e-4 \
#        --batch_size 10 \
#        --num_epoch 300 \
#        --ny 256 \
#        --nx 256 \
#        --nch 3 \
#        --nker 64 \
#        --wgt 1e2 \
#        --network 'pix2pix' \
#        --data_dir './../../datasets/facaes' \
#        --ckpt_dir './checkpoint' \
#        --log_dir './log' \
#        --result_dir './result'
#

python3 '/content/drive/My Drive/YouTube/pytorch-cyclegan/main.py' \
        --mode 'train' \
        --train_continue 'on' \
        --lr 2e-4 \
        --batch_size 4 \
        --num_epoch 100 \
        --ny 256 \
        --nx 256 \
        --nch 3 \
        --nker 64 \
        --wgt_cycle 1e2 \
        --wgt_ident 5e-1 \
        --network 'cyclegan' \
        --data_dir '/content/drive/My Drive/datasets/monet2photo' \
        --ckpt_dir '/content/drive/My Drive/YouTube/pytorch-cyclegan/checkpoint' \
        --log_dir '/content/drive/My Drive/YouTube/pytorch-cyclegan/log' \
        --result_dir '/content/drive/My Drive/YouTube/pytorch-cyclegan/result'

