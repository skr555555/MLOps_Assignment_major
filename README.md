# MLOps Major Assignment - g24ai2080
Note: Pipeline is from docker_cicd branch
1. Create environment
conda create -n mlops_env python=3.10
conda activate mlops_env
pip install -r requirements.txt

2. Train model
python train.py

3. Test model
python test.py

4. Run Flask App
python app.py

Docker Hub:

https://hub.docker.com/r/skr5/olivetti-decisiontree

Build Command
docker build -t skr5/olivetti-decisiontree:1.0 .

Run Command
docker run --rm -p 5000:5000 skr5/olivetti-decisiontree:1.0

Local URL
http://127.0.0.1:5000

Kubernetes Deployment:

Start Minikube
minikube start

Apply the Deployment
kubectl apply -f k8s-deployment.yaml

Verify 3 Pods Are Running
kubectl get pods


Verify Deployment Details
kubectl get deployments


Access the Flask App Running in Kubernetes

minikube service olivetti-service --url

