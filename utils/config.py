
import torch

class Config:
	def __init__(self, args) -> None:

		# data hyperparameter
		self.batch_size = args.batch_size
		self.train_num = args.train_num
		self.dev_num = args.dev_num
		self.test_num = args.test_num
		self.max_seq_length = args.max_seq_length
		### use as target as well for seq2seq model
		self.max_candidate_length = args.max_candidate_length if "max_candidate_length" in args.__dict__ else 20
		self.train_file = args.train_file
		self.dev_file = args.dev_file
		self.test_file = args.test_file
		self.gpus = args.gpus

		self.num_tagged_insts = args.num_tagged_insts if "num_tagged_insts" in args.__dict__ else None

		## only used for sequence to sequence model
		self.generated_max_length = args.generated_max_length if "generated_max_length" in args.__dict__ else 20

		# optimizer hyperparameter
		self.learning_rate = args.learning_rate
		self.max_grad_norm = args.max_grad_norm

		# training
		self.device = torch.device(args.device)
		self.num_epochs = args.num_epochs
		self.early_stop = args.early_stop
		self.num_workers = 8

		# model
		self.model_folder = args.model_folder
		self.bert_model_name = args.bert_model_name
		self.bert_folder = args.bert_folder
		self.parallel = args.parallel if "parallel" in args.__dict__ else 3
		self.sharing = args.sharing if "sharing" in args.__dict__ else "share-bart"
		self.grammar_loss = args.grammar_loss if "grammar_loss" in args.__dict__ else None
		self.num_labels = -1
		self.alpha = args.alpha if "alpha" in args.__dict__ else None
		self.add_candidate = args.add_candidate if "add_candidate" in args.__dict__ else None

		self.fp16 = args.fp16
