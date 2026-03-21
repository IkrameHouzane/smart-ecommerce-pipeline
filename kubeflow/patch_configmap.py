# kubeflow/patch_configmap.py
import subprocess
import json

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": "workflow-controller-configmap",
        "namespace": "kubeflow"
    },
    "data": {
        "executor": "image: quay.io/argoproj/argoexec:v3.4.17\nimagePullPolicy: Never\n",
        "artifactRepository": """archiveLogs: true
s3:
  endpoint: "minio-service.kubeflow:9000"
  bucket: "mlpipeline"
  keyFormat: "artifacts/{{workflow.name}}/{{workflow.creationTimestamp.Y}}/{{workflow.creationTimestamp.m}}/{{workflow.creationTimestamp.d}}/{{pod.name}}"
  insecure: true
  accessKeySecret:
    name: mlpipeline-minio-artifact
    key: accesskey
  secretKeySecret:
    name: mlpipeline-minio-artifact
    key: secretkey
"""
    }
}

with open("wf-configmap.json", "w") as f:
    json.dump(configmap, f)

result = subprocess.run(
    ["kubectl", "apply", "-f", "wf-configmap.json"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

subprocess.run(["kubectl", "rollout", "restart",
                "deployment/workflow-controller", "-n", "kubeflow"])
print("Done - workflow-controller restarted")