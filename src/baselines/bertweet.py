from comet_ml import Experiment
import os
import sys
sys.path.append("../data_processing")
import numpy as np
from loading import load_train_valid_data, load_test_data
import pandas as pd
import torch
from torch import nn, optim
import yaml

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import random
from transformers import AutoModel, AutoTokenizer

RANDOM_SEED = 285
EPOCHS = 5
BATCH_SIZE = 32
MAX_TOKEN_LENS = 100

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tweetNormalizer(sentence):
	sentence=sentence.replace("<url>", "HTTPURL")
	sentence=sentence.replace("<user>", "@USER")
	return sentence

def data_loader(df, tokenizer, max_len, batch_size):
	encoded_tweet = tokenizer.batch_encode_plus(
		df['tweet'].values,
		add_special_tokens=True,
		return_attention_mask=True, 
		pad_to_max_length=True, 
		max_length=max_len,
		return_tensors='pt'
	)
	input_ids = encoded_tweet['input_ids']
	attention_masks = encoded_tweet['attention_mask']
	# pytorch requires 0 ,1 for the label
	labels = [0 if i== -1 else 1 for i in df['label'].values]
	labels = torch.tensor(labels)
	dataset = TensorDataset(input_ids, attention_masks, labels)
	return DataLoader(
					dataset,
					batch_size=batch_size
		   )
def test_loader(test, tokenizer, max_len, batch_size):
	encoded_tweet = tokenizer.batch_encode_plus(
		test['tweet'].values,
		add_special_tokens=True,
		return_attention_mask=True, 
		pad_to_max_length=True, 
		max_length=max_len,
		return_tensors='pt'
	)
	input_ids = encoded_tweet['input_ids']
	attention_masks = encoded_tweet['attention_mask']
	dataset = TensorDataset(input_ids, attention_masks)
	return DataLoader(
					dataset,
					batch_size=batch_size
		   )

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		#transformer of choice
		self.transformer = AutoModel.from_pretrained("vinai/bertweet-base")
		
		#map to output
		self.dropout = nn.Dropout(p=0.3)
		self.dense = nn.Linear(self.transformer.config.hidden_size, 2)
		
	def forward(self, input_ids, attention_mask):
		 transformer_output = self.transformer(
									  input_ids=input_ids,
									  attention_mask=attention_mask
							)
		 dropout_output = self.dropout(transformer_output[1])
		 output = self.dense(dropout_output)
		 return output

def evaluate(model, valid, loss_fn):

	model.eval()
	
	total_loss = 0
	predictions, true = [], []
	with torch.no_grad():
		for batch in valid:

			batch = tuple(b.to(device) for b in batch)

			inputs = {'input_ids':      batch[0],
					  'attention_mask': batch[1],
					 }
			true_label = batch[2].to(device)
			
			outputs = model(**inputs)
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

			inputs = {'input_ids':      batch[0],
					  'attention_mask': batch[1],
					 }
			
			outputs = model(**inputs)
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

	experiment.set_name('bertweet (no-freeze with modified param)')

	params = {
		'random_seed': RANDOM_SEED,
		'model': 'bertweet',
		'train_size': BATCH_SIZE,
		'token_length': MAX_TOKEN_LENS,
		'EPOCHS': EPOCHS
	}
	experiment.log_parameters(params)
	path_to_dataset = os.path.join(os.pardir, os.pardir,  'dataset')
	train, valid = load_train_valid_data(path_to_dataset)
	test = load_test_data(path_to_dataset)
	print("load data")
	train['tweet'] = train['tweet'].apply(tweetNormalizer)
	valid['tweet'] = valid['tweet'].apply(tweetNormalizer)

	#tokenizer of choice
	tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

	train_loader = data_loader(train,tokenizer, MAX_TOKEN_LENS, BATCH_SIZE)
	valid_loader = data_loader(valid,tokenizer, MAX_TOKEN_LENS, BATCH_SIZE)
	print("load model")
	model = Model()
	model = model.to(device)


	optimizer = AdamW(
		model.parameters(),
		lr = 2e-5,
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
				inputs = {
					'input_ids': batch[0].to(device),
					'attention_mask': batch[1].to(device)
				}
				true_label = batch[2].to(device)
				outputs = model(**inputs)
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
				torch.save(model.state_dict(), 'bertweet_nofreeze.model')

			experiment.log_metric("train_epoch_loss", loss_train, epoch=epoch)   
			experiment.log_metric("val_epoch_loss", loss_val, epoch=epoch)    
			experiment.log_metric("val_epoch_accuracy", accuracy, epoch=epoch)


	# load Best model
	model.load_state_dict(torch.load("bertweet_nofreeze.model"))
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
	test['tweet'] = test['tweet'].apply(tweetNormalizer)
	test_tensor = test_loader(test, tokenizer, MAX_TOKEN_LENS,BATCH_SIZE)

	predicted_labels = predict(model, test_tensor)
	predicted_labels = [-1 if i ==0 else i for i in predicted_labels]
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
		filename = "bertweet_nofreeze_prediction.csv",
		tabular_data = predictions
	)
    predictions.to_csv("bertweet_nofreeze_prediction.csv")

if '__main__' == __name__:
	main()
