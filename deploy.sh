#!/usr/bin/env bash
set -e

# Kubernetes namespace
NAMESPACE="ship-detect"

echo "Applying Kubernetes manifests..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 898121064240.dkr.ecr.us-east-1.amazonaws.com
sudo kubectl apply -f ns.yaml 
sudo kubectl apply -f data-pvc.yaml
# sudo kubectl wait --for=condition=Bound pvc/ship-data-pvc -n ship-detect --timeout=300s
sudo kubectl apply -f model-pvc.yaml
# sudo kubectl wait --for=condition=Bound pvc/ship-model-pvc -n ship-detect --timeout=300s
sudo kubectl apply -f job.yaml
echo "Waiting for training job to complete..."
# Optionally, wait for job completion
sudo kubectl wait --for=condition=complete --timeout=1h job/ship-train-job -n $NAMESPACE

echo "Training job completed. Deploying inference service..."
sudo kubectl apply -f inference-dp.yaml
sudo kubectl apply -f inference-svc.yaml
sudo kubectl apply -f hpa.yaml

echo "Deployment initiated. You can monitor pods with 'kubectl get pods -n $NAMESPACE'."
echo "Once the LOAD BALANCER is ready, use 'kubectl get svc -n $NAMESPACE' to get the external IP/URL."
