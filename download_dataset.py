import kagglehub

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/mental-health")

print("Path to dataset files:", path)
