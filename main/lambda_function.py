try:
	import unzip_requirements
except ImportError:
	pass
from model import *
import json
import torch
# from torch.utils.data import DataLoader, Dataset
import time
import boto3

#"errorMessage": "An error occurred (RequestEntityTooLargeException) when calling the Invoke operation: 290312 byte payload is too large for the Event invocation type (limit 262144 bytes)"
    
def aggregate_grad(s3,num_workers):
    bucket='ratingdata291'
    pref ='grads/'
    ret = []
    num_added = 0
    iter_debug = 0
    # response = s3.list_objects_v2(Bucket=bucket,Prefix = pref)
    # print('count is', response['KeyCount']) 
    
    while (num_added < num_workers and iter_debug <40):
        response = s3.list_objects_v2(Bucket=bucket,Prefix = pref)
        # print('count is', response['KeyCount']) 
        if (response['KeyCount'] > 1):    #default 1 keycount(folder key), therefore we want >1 

            for c in response['Contents']:
                k =c['Key']
                # print('key name is :', k)
                if (k!= pref and k.endswith('json')):
                    print('key name is :', k)
                    obj = s3.get_object(Bucket=bucket, Key =k)
                    file_content = obj['Body'].read().decode('utf-8')
                    json_content = json.loads(file_content)
                    grad_arr = []
                    for x in json_content:
                        t = torch.FloatTensor(x)
                        grad_arr.append(t)
                    ret.append(grad_arr)
                    print(k, ' has been added')
                    s3.delete_object(Bucket = bucket, Key =k)  #delete object afterwards.   
                    num_added+=1
            if num_added == num_workers:
                break

        time.sleep(2)
        iter_debug+=1

    if (iter_debug >= 10):
        print('iter exceeded. DEBUG')
    return ret
    

def save_weights(s3, generator, filename):
    grads = []
    for x in generator:
        grads.append(x.tolist())
        
    content = json.dumps(grads)
    
    s3.put_object(
        Bucket = 'ratingdata291',
        Key = filename,
        Body = content
    )
    print('Weights saved successfully.')

        
      
        
def lambda_handler(event, context):
    
    client = boto3.client('lambda')
    s3 = boto3.client('s3')
    num_workers = 5
    
    #set model parameters for all workers 
    params = {'num_users': 6041, 
          'num_items': 3707, 
          'global_mean': 3, 
          'latent_dim': 3
    }
    
    #Create main Model, instantiate with parameters
    bucket='ratingdata291'
    model = BiasMF(params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # save_weights(model.state_dict())
    save_weights(s3, model.parameters(), 'model_weights.json')


    
    #Specify hyperparams: 
    num_epoch = 5
    workers = [ 'arn:aws:lambda:us-east-1:949216546098:function:childWorker1','arn:aws:lambda:us-east-1:949216546098:function:childWorker2','arn:aws:lambda:us-east-1:949216546098:function:childWorker3','arn:aws:lambda:us-east-1:949216546098:function:childWorker4','arn:aws:lambda:us-east-1:949216546098:function:childWorker5']  

    #train for n epochs
    start_time = time.time()
    for epoch in range(num_epoch):
        #prepare param to be passed into child worker functions
        inputParams = {}
        inputParams['model_params'] = params
        
        for worker in workers:  #each child executes async. 
            response = client.invoke(
                FunctionName = worker,
                InvocationType = 'Event',
                Payload = json.dumps(inputParams)  #could pass in the model, lets try. //currently only use model_params
            )

            
        grad_arrs = aggregate_grad(s3,3)     #return a list of grads from each worker 

        
        #zero grad
        for param in model.parameters():
            if not (param.grad is None):
                param.grad *=0
        
        #update main model parameter.grad
        i = 0
        for param in model.parameters():
            for j in range(num_workers):
                tsr = torch.FloatTensor(grad_arrs[j][i])
                if (param.grad is None):     
                    param.grad = tsr
                else:
                    param.grad += tsr
            i+=1
            
        
        #step and update weights (backprop)
        optimizer.step()
        
        #save weights to local file
        save_weights(s3, model.parameters(), 'model_weights.json')
        print('\n')
        print('epoch i finished at %s seconds.',(time.time() - start_time))


    return {
        'statusCode': 200,
        'body': json.dumps('Hello from main!')
    }
