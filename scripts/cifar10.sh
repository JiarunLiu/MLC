for seed in 1 2 3 4 5
do
  python MLC.py --dir experiment/cifar10/mlc/sn_0.2/trial_${seed} \
   --dataset 'cifar10' --noise_type sn --noise 0.2 --optim "Adam" \
   --forget-rate 0.2 --lambda1 3000 --backbone 'cnn' --lr 5e-3 --lr2 1e-4 --random-seed ${seed}

  python MLC.py --dir experiment/cifar10/mlc/sn_0.4/trial_${seed} \
   --dataset 'cifar10' --noise_type sn --noise 0.4 --optim "Adam" \
   --forget-rate 0.4 --lambda1 4000 --backbone 'cnn' --lr 5e-3 --lr2 1e-4 --random-seed ${seed}

  python MLC.py --dir experiment/cifar10/mlc/sn_0.8/trial_${seed} \
   --dataset 'cifar10' --noise_type sn --noise 0.8 --optim "Adam" \
   --forget-rate 0.2 --lambda1 3000 --backbone 'cnn' --lr 5e-3 --lr2 1e-4 --random-seed ${seed}

  python MLC.py --dir experiment/cifar10/mlc/pair_0.2/trial_${seed} \
   --dataset 'cifar10' --noise_type pairflip --noise 0.2 --optim "Adam" \
   --forget-rate 0.2 --lambda1 1500 --backbone 'cnn' --lr 5e-3 --lr2 1e-4 --random-seed ${seed}

  python MLC.py --dir experiment/cifar10/mlc/pair_0.45/trial_${seed} \
   --dataset 'cifar10' --noise_type pairflip --noise 0.45 --optim "Adam" \
   --forget-rate 0.2 --lambda1 4000 --backbone 'cnn' --lr 5e-3 --lr2 1e-4 --random-seed ${seed}
done