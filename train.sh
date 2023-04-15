python train.py --train --algorithm=SAC --reward=euclidean
python train.py --train --algorithm=PPO --reward=euclidean
python train.py --train --algorithm=DDPG --reward=euclidean

python train.py --train --algorithm=SAC --reward=euclidean_bar
python train.py --train --algorithm=PPO --reward=euclidean_bar
python train.py --train --algorithm=DDPG --reward=euclidean_bar

python train.py --train --algorithm=SAC --reward=square
python train.py --train --algorithm=PPO --reward=square
python train.py --train --algorithm=DDPG --reward=square

python train.py --train --algorithm=SAC --reward=square_bar
python train.py --train --algorithm=PPO --reward=square_bar
python train.py --train --algorithm=DDPG --reward=square_bar