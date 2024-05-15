
By running the code from the Diabetes model to be deployed.ipynb notebook, you will create a pickel file in the trained_model folder.

Assuming the abouve is done! 

**Follow the below steps::**

docker login 
usrname and password

docker build -t yourusername/fastapi-ml-app .

docker tag yourusername/fastapi-ml-app:latest yourusername/fastapi-ml-app:latest

docker push yourusername/fastapi-ml-app:latest

minikube stop

minikube start

kubectl create deployment fastapi-ml-app --image=yourusername/fastapi-ml-app:latest

kubectl expose deployment fastapi-ml-app --type=LoadBalancer --port=8009

minikube service fastapi-ml-app --url


kubectl get pods

kubectl cluster-info


docker logout 
