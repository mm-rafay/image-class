sudo kubectl delete pvc ship-model-pvc -n ship-detect
sudo kubectl delete pvc ship-data-pvc -n ship-detect
sudo kubectl delete job ship-train-job -n ship-detect
sudo kubectl delete ns ship-detect
