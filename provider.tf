terraform {
  required_providers {
     rafay = {
      version = ">= 0.1"
      source = "registry.terraform.io/RafaySystems/rafay"
    }
  }
}

provider "rafay" {
    provider_config_file = "/home/terraform/app/scratch/job/rafay-config.json"
}
