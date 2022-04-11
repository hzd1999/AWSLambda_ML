try:
	import unzip_requirements
except ImportError:
	pass
import json
import torch
# from torch.utils.data import DataLoader, Dataset
import boto3
import csv
import numpy as np
from model import *

def load_weights(s3,model):
    #load weights and update 
    bucket = 'ratingdata291'
    k = 'model_weights.json'
    obj = s3.get_object(Bucket=bucket, Key =k)
    file_content = obj['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    i =0
    for param in model.parameters():
        param.data = torch.FloatTensor(json_content[i])
        param.requires_grad = True
        i+=1
    print('weights loaded successfully')
    
def save_grad(s3, generator, filename):
    grads = []
    for x in generator:
        grads.append(x.grad.tolist())
        
    content = json.dumps(grads)
    
    key = 'grads/'+filename
    s3.put_object(
        Bucket = 'ratingdata291',
        Key = key,
        Body = content
    )
    print('saved successfully.')
    
    
def read_data(key, bucket,s3):
    original = s3.get_object(
    Bucket=bucket, 
    Key=key)
    
    data = original['Body'].read()
    data = data.decode('utf-8').splitlines()
    lines = list(csv.reader(data))
    lines = np.array(lines)[1:].astype(int)
    users = lines[:,4]
    items =lines[:,5]
    ratings = lines[:,2]
    return users,items,ratings


    
def lambda_handler(event, context):
    # 1 Read the input parameters
    
    
    client = boto3.client('lambda')
    s3 = boto3.client('s3')
    
    users,items,ratings = read_data('d1.csv','ratingdata291',s3)


    user_tensor = torch.LongTensor(users)
    item_tensor = torch.LongTensor(items)
    rating_tensor = torch.FloatTensor(ratings)
    
    u, i, r = user_tensor,item_tensor,rating_tensor
    
    params = event['model_params']
  
    model = BiasMF(params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    load_weights(s3, model)                              #load model from main model
    #load model from main model
    #WRITE HERE

    
    #computes gradient for one epoch with the given data
    preds = model(u, i)  #forward pass
    loss = criterion(preds, r)   #compute loss
    optimizer.zero_grad()    #zero grad
    loss.backward()      #compute gradient using backprop
    save_grad(s3,model.parameters(), 'grad_w1.json')     #save gradients



 

    return {
        'statusCode': 200,
        'body': 'worker1 successful'
    }
