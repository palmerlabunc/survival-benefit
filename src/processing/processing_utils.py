import yaml

def load_config():
    file_path = 'config.yaml'
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)