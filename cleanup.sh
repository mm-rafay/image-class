sudo kubectl delete pvc image-model-pvc -n image-class
sudo kubectl delete pvc image-data-pvc -n image-class
sudo kubectl delete job image-train-job -n image-class
sudo kubectl delete ns image-class
sudo kubectl delete secret ecr-creds
