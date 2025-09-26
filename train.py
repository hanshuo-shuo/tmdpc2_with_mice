import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer, PTSDBuffer, PTSDBuffer2
from envs import make_prey_env
from tdmpc2 import TDMPC2
from tdmpc_surprise import TDMPC_surprise
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	buffer = Buffer(cfg)

	trainer = OnlineTrainer(
		cfg=cfg,
		env=make_prey_env(cfg),
		agent=TDMPC2(cfg),
		buffer=buffer,
		logger=Logger(cfg),
	)
	# trainer = OnlineTrainer(
	# 	cfg=cfg,
	# 	env=make_prey_env(cfg),
	# 	agent=TDMPC2(cfg),
	# 	buffer=PTSDBuffer(cfg),
	# 	logger=Logger(cfg),
	# )
	trainer.train()
	print('\nTraining completed')


if __name__ == '__main__':
	train()