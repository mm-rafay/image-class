#!/usr/bin/env bash
set -e

# Set AWS details
AWS_ACCOUNT_ID=""
AWS_REGION="us-east-1"         # e.g., us-east-1
TRAIN_REPO="ship-train"
INF_REPO="ship-inference"

# ECR repository URIs
TRAIN_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$TRAIN_REPO:latest"
INF_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$INF_REPO:latest"

# Authenticate Docker to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push training image
echo "Building training image..."
docker build -f Dockerfile.train -t $TRAIN_IMAGE_URI .
echo "Pushing training image to ECR..."
docker push $TRAIN_IMAGE_URI

# Build and push inference image
echo "Building inference image..."
docker build -f Dockerfile.inference -t $INF_IMAGE_URI .
echo "Pushing inference image to ECR..."
docker push $INF_IMAGE_URI

echo "Docker images for training and inference have been pushed to ECR."
