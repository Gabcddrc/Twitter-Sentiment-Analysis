from collections import defaultdict
from comet_ml import Experiment
import os
import sys
sys.path.append("../data_processing")
import numpy as np
from loading import load_train_valid_data, load_test_data
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import yaml

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import random

RANDOM_SEED = 285
EPOCHS = 20
BATCH_SIZE = 32

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GloVe embeddings
glove_dim = 200
glove_path = os.path.join(
  os.path.pardir,
  os.path.pardir,
  'embeddings',
  'glove.twitter.27B',
  'glove.twitter.27B.' + str(glove_dim) + 'd.txt'
)
glove_embeddings = defaultdict(lambda: None)
with open(glove_path, 'r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype=np.float64)
    glove_embeddings[word] = vector

# Tweet encoder returning average GloVe embedding
def encode_tweet(tweet):
  tokens = tweet.split()
  tokens = [glove_embeddings[w] for w in tokens]
  tokens = [w for w in tokens if w is not None]
  if len(tokens) > 0:
  	encoding = np.mean(tokens, axis=0)
  else:
	  encoding = np.zeros_like(glove_embeddings['<user>'])
  return encoding

def data_loader(df, batch_size):
	tweets = df['tweet'].values
	encoded_tweets = [encode_tweet(tweet) for tweet in tweets]
	encoded_tweets = np.array(encoded_tweets, dtype='float32')
	inputs = torch.tensor(encoded_tweets)
	labels = [0 if i== -1 else 1 for i in df['label'].values]
	labels = torch.tensor(labels)
	dataset = TensorDataset(inputs, labels)
	return DataLoader(dataset, batch_size=batch_size)

def test_loader(test, batch_size):
	tweets = test['tweet'].values
	encoded_tweets = [encode_tweet(tweet) for tweet in tweets]
	encoded_tweets = np.array(encoded_tweets, dtype='float32')
	inputs = torch.tensor(encoded_tweets)
	dataset = TensorDataset(inputs)
	return DataLoader(dataset, batch_size=batch_size)

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.hidden0 = nn.Linear(glove_dim, 512)
		self.hidden1 = nn.Linear(512, 512)
		self.output = nn.Linear(512, 2)

	def forward(self, inputs):
		h = F.relu(self.hidden0(inputs))
		h = F.relu(self.hidden1(h))
		output = self.output(h)
		return output

def evaluate(model, valid, loss_fn):

	model.eval()
	
	total_loss = 0
	predictions, true = [], []
	with torch.no_grad():
		for batch in valid:

			batch = tuple(b.to(device) for b in batch)
			inputs = batch[0]
			true_label = batch[1]
			
			outputs = model(inputs)
			_, prediction = torch.max(outputs, dim=1)
			
			loss = loss_fn(outputs, true_label)
			total_loss +=loss.item()
			
			predictions.append(prediction.detach().cpu().numpy())
			true.append(true_label.cpu().numpy())
	
	loss = total_loss/len(valid) 
	
	predictions = np.concatenate(predictions, axis=0)
	true = np.concatenate(true, axis=0)
			
	return loss, accuracy_score(predictions, true)

def predict(model, valid):

	model.eval()
	
	predictions = []
	with torch.no_grad():
		for batch in valid:

			batch = tuple(b.to(device) for b in batch)
			inputs = batch[0]

			outputs = model(inputs)
			_, prediction = torch.max(outputs, dim=1)
			
			
			predictions.append(prediction.detach().cpu().numpy())
	
	predictions = np.concatenate(predictions, axis=0)
	return predictions

def main():
	experiment = Experiment(
		api_key="",
		project_name="cil-project",
		workspace="smueksch",
		auto_param_logging=False,
        disabled=True,
	)

	experiment.set_name('GloVe MLP (average embedding)')

	params = {
		'random_seed': RANDOM_SEED,
		'model': 'glove_mlp',
		'train_size': BATCH_SIZE,
		'EPOCHS': EPOCHS
	}
	experiment.log_parameters(params)
	path_to_dataset = os.path.join(os.pardir, os.pardir,  'dataset')
	train, valid = load_train_valid_data(path_to_dataset)
	test = load_test_data(path_to_dataset)
	print("load data")
	train['tweet'] = train['tweet']
	valid['tweet'] = valid['tweet']

	train_loader = data_loader(train, BATCH_SIZE)
	valid_loader = data_loader(valid, BATCH_SIZE)
	print("load model")
	model = Model()
	model = model.to(device)

	optimizer = AdamW(
		model.parameters(),
		lr = 1e-4,
		correct_bias = False,
	)

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps = len(train_loader)*EPOCHS
	)
	loss_fn = nn.CrossEntropyLoss().to(device)

	#training
	with experiment.train():

		best_accuracy = 0
		step = 0
		for epoch in range(1, EPOCHS+1):
			model.train()
			total_loss_train = 0

			for batch in train_loader:
				model.zero_grad()
				inputs = batch[0].to(device)
				true_label = batch[1].to(device)
				outputs = model(inputs)
				_, prediction = torch.max(outputs, dim=1)
				loss = loss_fn(outputs, true_label)
				total_loss_train +=loss.item()
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
				scheduler.step()
				step += 1
				experiment.log_metric("batch_loss", loss.item()/len(batch), step=step)

			loss_train = total_loss_train/len(train_loader)

			loss_val, accuracy = evaluate(model, valid_loader, loss_fn)

			#save best model
			if(accuracy > best_accuracy):
				print("new best accuracy")
				best_accuracy = accuracy
				torch.save(model.state_dict(), 'glove_average_mlp.model')

			experiment.log_metric("train_epoch_loss", loss_train, epoch=epoch)   
			experiment.log_metric("val_epoch_loss", loss_val, epoch=epoch)    
			experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch)


	# load Best model
	model.load_state_dict(torch.load("glove_average_mlp.model"))
	loss_val, train_accuracy = evaluate(model, train_loader, loss_fn)
	loss_val, val_accuracy = evaluate(model, valid_loader, loss_fn)

	metrics = {
		'train_score': float(train_accuracy),
		'valid_score': float(val_accuracy)
	}
	experiment.log_metrics(metrics)
    # Log metrics to local file system
    with open('metrics.yaml', 'w+') as outfile:
        yaml.dump(metrics, outfile)


	#predict test dataset
	test['tweet'] = test['tweet']
	test_tensor = test_loader(test, BATCH_SIZE)

	predicted_labels = predict(model, test_tensor)
	predicted_labels = [-1 if i == 0 else i for i in predicted_labels]
	predictions = pd.DataFrame(
		predicted_labels,
		index = test.index,
		columns = ['Prediction']
	)

	# Adjust ID column name to fit the format expected in the output.
	predictions.index.name = 'Id'

	# Log predictions as a CSV to CometML, retrievable under the experiment
	# by going to `Assets > dataframes`.
	experiment.log_table(
		filename = "glove_average_mlp_prediction.csv",
		tabular_data = predictions
	)
    predictions.to_csv("glove_average_mlp_prediction.csv")

if '__main__' == __name__:
	main()
