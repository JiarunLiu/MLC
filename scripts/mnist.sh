for seed in 1 2 3 4 5
do
  python MLC.py --dir experiment/mnist/mlc/sn_0.2/trial_${seed} \
   --dataset 'mnist' --noise_type sn --noise 0.2 \
   --optim "Adam" --discard  0 --forget-rate 0.2 --lambda1 1000 \
   --stage1 30 --stage2 140 --epochs 320 --backbone mlp --random-seed ${seed} &

  python MLC.py --dir experiment/mnist/mlc/sn_0.4/trial_${seed} \
   --dataset 'mnist' --noise_type sn --noise 0.4 \
   --optim "Adam" --forget-rate 0.4 --lambda1 3000 \
   --stage1 30 --stage2 140 --epochs 320 --backbone mlp --random-seed ${seed} &

  python MLC.py --dir experiment/mnist/mlc/sn_0.8/trial_${seed} \
   --dataset 'mnist' --noise_type sn --noise 0.8 \
   --optim "Adam" --forget-rate 0.8 --lambda1 3000 \
   --stage1 30 --stage2 140 --epochs 320 --backbone mlp --random-seed ${seed} &

  python MLC.py --dir experiment/mnist/mlc/pair_0.2/trial_${seed} \
   --dataset 'mnist' --noise_type pairflip --noise 0.2 \
   --optim "Adam" --forget-rate 0.2 --lambda1 2000 \
   --stage1 30 --stage2 140 --epochs 320 --backbone mlp --random-seed ${seed} &

  python MLC.py --dir experiment/mnist/mlc/pair_0.45/trial_${seed} \
   --dataset 'mnist' --noise_type pairflip --noise 0.45 \
   --optim "Adam" --forget-rate 0.45 --lambda1 2500 \
   --stage1 50 --stage2 140 --epochs 320 --backbone mlp --random-seed ${seed} &
done
