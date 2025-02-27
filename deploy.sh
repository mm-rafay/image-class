#!/usr/bin/env bash
set -e

# Kubernetes namespace
NAMESPACE="ship-detect"

echo "Applying Kubernetes manifests..."
kubectl apply -f k8s-namespace-pvc.yaml      # contains Namespace and PVCs
kubectl apply -f k8s-train-job.yaml          # Training Job manifest
echo "Waiting for training job to complete..."
# Optionally, wait for job completion
kubectl wait --for=condition=complete --timeout=1h job/ship-train-job -n $NAMESPACE

echo "Training job completed. Deploying inference service..."
kubectl apply -f k8s-inference-deployment.yaml
kubectl apply -f k8s-inference-service.yaml

echo "Deployment initiated. You can monitor pods with 'kubectl get pods -n $NAMESPACE'."
echo "Once the LOAD BALANCER is ready, use 'kubectl get svc -n $NAMESPACE' to get the external IP/URL."
