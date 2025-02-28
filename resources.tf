resource "null_resource" "deploy" {
  provisioner "local-exec" {
    command = "echo "Deploying..."
  }
}
