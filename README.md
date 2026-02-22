# EGT307_TYPE_M_Assignment

# Predictive Maintenance System with Microservices Architecture
## Project Overview and Objectives

This project implements a scalable Predictive Maintenance System designed to forecast equipment failures using machine learning. The system is built with a microservices architecture and deployed using Docker and Kubernetes to ensure flexibility, scalability, and reliability.

The primary objective of this project is to demonstrate how modern cloud-native technologies can be integrated with machine learning workflows to support real-time predictions, batch processing, and continuous model retraining.

Key objectives include:

Develop a machine learning model capable of predicting equipment failures.

Design independent microservices to support modular system architecture.

Enable real-time and batch predictions through API endpoints.

Support automated model retraining using newly uploaded datasets.

Deploy the system using Docker for containerisation and Kubernetes for orchestration.

Ensure system logs and prediction records are persistently stored.

## Instructions to Build, Run, and Deploy the System
Step 1: Clone the Repository

git clone https://github.com/TANKK06/EGT307_TYPE_M_Assignment.git

cd <repository-folder>


### run locally

1) Start Minikube

minikube delete

minikube start

2) Enable addons (Ingress + Metrics for HPA)

minikube addons enable ingress

minikube addons enable metrics-server

3) Build Docker images INSIDE Minikube#   (so K8s can use them without docker push)

eval $(minikube -p minikube docker-env)

docker build -t docker.io/kaykit/pm-logger:v1 ./services/logger

docker build -t docker.io/kaykit/pm-inference:v1 ./services/inference-api

docker build -t docker.io/kaykit/pm-dashboard:v1 ./services/dashboard

docker build -t docker.io/kaykit/pm-batch-predict:v1 -f services/batch-predict/Dockerfile .

docker build -t docker.io/kaykit/pm-trainer:v1 -f services/trainer/Dockerfile .

4) Apply Kubernetes manifests

kubectl apply -f k8s/namespace.yaml

kubectl apply -f k8s/

5) Watch your pods come up

kubectl get pods -n pm -w

6) Check services + ingress

kubectl get svc -n pm

kubectl get ingress -n pm

kubectl get hpa -n pm

7) Verify Metrics Server works (for HPA)

kubectl top nodes

kubectl top pods -n pm

8) Open the Dashboard

minikube service -n pm dashboard --url

### Docker Hub

1) Start Minikube

minikube delete

minikube start

2) Enable addons (Ingress + Metrics for HPA)

minikube addons enable ingress

minikube addons enable metrics-server

4) Apply Kubernetes manifests

kubectl apply -f k8s/namespace.yaml

kubectl apply -f k8s/  

5) Watch your pods come up

kubectl get pods -n pm -w

When all the pods are running them ctrl C

6) Open the Dashboard

minikube service -n pm dashboard --url

## Description of Each Microservice

### Dashboard Service
Provides a user-friendly interface that allows users to:

Submit data for real-time prediction

Upload datasets for batch prediction

Trigger model retraining

View prediction results

This service acts as the primary interaction layer between the user and the backend system.

### Inference API
Handles prediction requests by loading the latest trained machine learning model from persistent storage.

Core responsibilities:

Serve real-time prediction requests

Reload the model after retraining

Ensure low-latency responses

### Trainer Service
Responsible for training and updating the machine learning model.

Key functions:

Preprocess incoming datasets

Train and evaluate the predictive model

Save the best-performing model

Notify the inference service to reload the updated model

### Batch Prediction Service
Processes large datasets asynchronously and generates prediction outputs for offline analysis.
This service is useful for organisations that require periodic evaluation of equipment health across multiple machines.

### Logger Service
Captures and stores important system data, including:

Prediction requests

Prediction results

Training activity

Persistent logging improves traceability, debugging, and auditability.

## Dataset Information and Sources

### Machine Predictive Maintenance Classification Dataset
Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, we present and provide a synthetic dataset that reflects real predictive maintenance encountered in the industry to the best of our knowledge.

The dataset consists of 10 000 data points stored as rows with 14 features in columns

UID: unique identifier ranging from 1 to 10000

productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number

air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K

process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.

rotational speed [rpm]: calculated from powepower of 2860 W, overlaid with a normally distributed noise

torque [Nm]: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.

tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a

'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.

Important : There are two Targets - Do not make the mistake of using one of them as feature, as it will lead to leakage.

Target : Failure or Not

Failure Type : Type of Failure

### Dataset Source:
AI4I 2020 Predictive Maintenance Dataset

https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset

## Purpose of Kubernetes in This System
1. Container Orchestration

Kubernetes manages all microservices:

dashboard

inference API

batch prediction

trainer

logger

PostgreSQL

It ensures each service runs correctly and communicates through internal networking.

Without Kubernetes: services must be started manually.

With Kubernetes: services run automatically and consistently.

2. Automatic Recovery (Self-Healing)
If a service crashes:

Kubernetes automatically restarts the pod

system continues running without manual intervention

This improves system reliability and uptime.

3. Scalability & Load Handling
Kubernetes enables Horizontal Pod Autoscaling (HPA):

increases pods when traffic increases

reduces pods when load is low

Example:

heavy batch prediction → inference service scales up

low usage → scales down to save resources

This ensures efficient resource usage and performance.

4. Persistent Storage for Models & Data
A Persistent Volume Claim (PVC) is used to store:

trained ML models

model artifacts

This ensures the models are not lost when pods restart and inference always loads the latest trained model

5. Service Discovery & Internal Networking
Kubernetes provides built-in networking:

services communicate using service names

no need to manage IP addresses

Example:

trainer → inference reload request

inference → logger service

dashboard → all services

This simplifies microservice communication.

6. External Access via Ingress
Ingress allows external users to access the system through a single entry point.

Benefits:

clean routing

easier access management

production-like architecture

7. Production-Like Deployment Environment
Using Kubernetes simulates real-world deployment environments used in industry.

This improves:

deployment reliability

scalability testing

fault tolerance

maintainability

## Known Issues and Limitations

Although the system is fully functional, several limitations exist:

1. Image Caching in Kubernetes

When using imagePullPolicy: IfNotPresent, Kubernetes may reuse cached images even after updates. It is recommended to use versioned image tags (e.g., v1, v2) to prevent deployment inconsistencies.

2. Dependency Consistency

Trainer and inference services must use identical library versions. Differences may cause the inference service to fail when loading newly trained models.

3. Resource Constraints in Minikube

Minikube runs on local hardware and may experience performance limitations when handling large datasets or concurrent requests.

4. Manual Retraining Trigger

Model retraining currently requires manual initiation. Future improvements could include automated retraining pipelines.

5. Security Enhancements

Authentication and role-based access control are not implemented and should be added before production deployment.
