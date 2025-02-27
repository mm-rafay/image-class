#!/usr/bin/env bash
set -e

# Kubernetes namespace
NAMESPACE="ship-detect"

echo "Applying Kubernetes manifests..."
sudo kubectl apply -f ns.yaml 
sudo kubectl apply -f data-pvc.yaml
sudo kubectl apply -f model-pvc.yaml
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
