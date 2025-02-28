#!/usr/bin/env bash
set -e

NAMESPACE="ship-detect"

echo "Applying Kubernetes manifests..."

# 1. Create the namespace (if not already existing)
kubectl apply -f ns.yaml

# 2. Create an ECR pull secret
kubectl create secret docker-registry ecr-creds \
    --namespace "$NAMESPACE" \
    --docker-server=898121064240.dkr.ecr.us-east-1.amazonaws.com \
    --docker-username=AWS \
    --docker-password="$(aws ecr get-login-password --region us-east-1)" \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. Apply the data PVC first (where your dataset will live)
kubectl apply -f data-pvc.yaml

# 4. Launch the data loader job (downloads Kaggle dataset -> /mnt/data)
kubectl apply -f data-loader.yaml

echo "Waiting for data loader job to complete..."
kubectl wait --for=condition=complete --timeout=30m job/ship-data-loader -n "$NAMESPACE"

# 5. Now that data is in the PVC, apply model PVC, then training job
kubectl apply -f model-pvc.yaml
kubectl apply -f job.yaml

echo "Waiting for training job to complete..."
kubectl wait --for=condition=complete --timeout=1h job/ship-train-job -n "$NAMESPACE"

# 6. Deploy inference
echo "Training job completed. Deploying inference service..."
kubectl apply -f inference-dp.yaml
kubectl apply -f inference-svc.yaml
kubectl apply -f hpa.yaml

echo "Deployment initiated. Monitor with 'kubectl get pods -n $NAMESPACE'."
