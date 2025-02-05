terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
  credentials = file(var.credentials)
  project     = var.project
  region      = var.region
}

data "google_project" "project" {
}

resource "google_cloud_run_service" "classify-api" {
  name     = "chess-opening-classification-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project}/chess-opening-classification-api:latest"
        ports {
          container_port = 8000
        }
      }
    }

     metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1" 
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  metadata {
    annotations = {
      "run.googleapis.com/ingress" = "all"
    }
  }

}

resource "google_cloud_run_service_iam_binding" "classify-api-invoker" {
  location = google_cloud_run_service.classify-api.location
  service  = google_cloud_run_service.classify-api.name
  role     = "roles/run.invoker"
  members = [
    "allUsers"
  ]
}


output "classify-api" {
  value = google_cloud_run_service.classify-api.status[0].url
}