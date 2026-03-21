# kubeflow/fix_yaml.py
yaml_path = 'smart_ecommerce_pipeline.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = "image: smart-ecommerce-pipeline:local"
new = "image: smart-ecommerce-pipeline:local\n        imagePullPolicy: Never"

content = content.replace(old, new)

with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done:', content.count('imagePullPolicy'), 'occurrences')