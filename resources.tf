resource "null_resource" "deploy" {
  provisioner "local-exec" {
    command = "echo Deploying..."
  }
}

/* resource "rafay_workload" "ns" {
  metadata {
    name    = "ns"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://ns.yaml"
        }
      }
    }
  }
}

resource "rafay_workload" "data-pvc" {
  metadata {
    name    = "data-pvc"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://data-pvc.yaml"
        }
      }
    }
  }
}

resource "rafay_workload" "model-pvc" {
  metadata {
    name    = "model-pvc"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://model-pvc.yaml"
        }
      }
    }
  }
}

resource "rafay_workload" "job" {
  metadata {
    name    = "job"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://job.yaml"
        }
      }
    }
  }
} */

resource "rafay_workload" "inference-dp" {
  metadata {
    name    = "inference-dp"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://inference-dp.yaml"
        }
      }
    }
  }
}

resource "rafay_workload" "inference-svc" {
  metadata {
    name    = "inference-svc"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://inference-svc.yaml"
        }
      }
    }
  }
}

resource "rafay_workload" "hpa" {
  metadata {
    name    = "hpa"
    project = "gpu-paas-demo"
  }
  spec {
    namespace = "default"
    placement {
      selector = "rafay.dev/clusterName=mm-gpu"
    }
    version = "v0"
    artifact {
      type = "Yaml"
      artifact {
        paths {
          name = "file://hpa.yaml"
        }
      }
    }
  }
}
