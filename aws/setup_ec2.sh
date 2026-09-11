#!/bin/bash
# AWS EC2 Setup Script (Ubuntu)
# Sets up the environment and starts the Flask server.

echo "Updating system..."
sudo apt-get update && sudo apt-get upgrade -y

echo "Installing dependencies..."
sudo apt-get install -y python3-pip python3-venv git nginx libgl1-mesa-glx

echo "Setting up app directory..."
mkdir -p /home/ubuntu/skin_cancer_app
cd /home/ubuntu/skin_cancer_app

# Normally, here you would git clone the repository:
# git clone <your-repo-url> .

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python packages..."
# Install packages
pip install -r requirements.txt
pip install gunicorn

echo "Setting up systemd service for Gunicorn..."
cat << 'EOF' | sudo tee /etc/systemd/system/skincancer.service
[Unit]
Description=Gunicorn instance to serve Skin Cancer Diagnosis App
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/skin_cancer_app
Environment="PATH=/home/ubuntu/skin_cancer_app/venv/bin"
ExecStart=/home/ubuntu/skin_cancer_app/venv/bin/gunicorn --workers 3 --bind unix:skincancer.sock -m 007 web.app:app

[Install]
WantedBy=multi-user.target
EOF

echo "Starting Gunicorn service..."
sudo systemctl start skincancer
sudo systemctl enable skincancer

echo "Configuring Nginx..."
cat << 'EOF' | sudo tee /etc/nginx/sites-available/skincancer
server {
    listen 80;
    server_name _;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/skin_cancer_app/skincancer.sock;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/skincancer /etc/nginx/sites-enabled
sudo systemctl restart nginx

echo "Deployment complete! App should be running on Port 80."
