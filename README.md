# AWSLambda_ML
Training Recommender Systems using Serverless Architecture with Data Parallelism

The advent of serverless computing has brought convenience and efficiency to many applications with its lightweight management and pricing model. As platform like AWS Lambda is gradually increasing its capacities with larger memory and greater package support, recent research have started to experiment serverless architecture with more data-intensive workloads such as Machine Learning training. In this project, I ported a ML-based Recommender System workload onto AWS Lambda and profiled its resource consumption. Using the concurrency of Lambda functions, we re-implemented the program with data parallelism and improved the training speed by 51\% without significant overhead. 
