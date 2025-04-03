#!/bin/bash

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    log_error "Please run as root (use sudo)"
    exit 1
fi

# Check Ubuntu version
if ! grep -q "Ubuntu 22.04" /etc/os-release; then
    log_warn "This script is tested on Ubuntu 22.04. Your system may not be fully compatible."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Add Coral repository with modern keyring approach
log_info "Adding Coral repository..."
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Install system dependencies
log_info "Installing system dependencies..."
apt update
apt install -y \
    git \
    python3-pip \
    python3-venv \
    libedgetpu1-std \
    python3-opencv \
    usbutils \
    pciutils \
    curl \
    wget \
    pkg-config \
    libsystemd-dev \
    linux-modules-extra-$(uname -r)

# Install Edge TPU runtime
log_info "Installing Edge TPU runtime..."
apt update
apt install -y \
    libedgetpu1-std \
    gasket-dkms \
    linux-modules-extra-$(uname -r)

# Reload modules
modprobe gasket
modprobe apex

# Verify Edge TPU is accessible
log_info "Checking for Edge TPU device..."
if ! lspci | grep -iE "Edge TPU|Coral|Global Unichip.*TPU" > /dev/null; then
    if ! lsusb | grep -iE "Edge TPU|Coral|Global Unichip" > /dev/null; then
        log_error "No Edge TPU device found"
        log_info "Available PCI devices:"
        lspci
        log_info "Available USB devices:"
        lsusb
        exit 1
    fi
fi

log_info "Edge TPU device found"

# Create directories
log_info "Creating directories..."
mkdir -p /opt/reef_server
mkdir -p /var/log/reef

# Set permissions
chown $SUDO_USER:$SUDO_USER /opt/reef_server
chown $SUDO_USER:$SUDO_USER /var/log/reef
chmod 755 /opt/reef_server
chmod 755 /var/log/reef

# Ask user about repository setup
if [ -d "/opt/reef_server" ]; then
    log_info "Reef server directory already exists at /opt/reef_server"
    read -p "Would you like to (s)kip, (o)verwrite with git clone, or (c)opy current files? [s/o/c]: " -n 1 -r
    echo
    case $REPLY in
        [Oo]* )
            log_info "Removing existing directory and cloning fresh copy..."
            rm -rf /opt/reef_server
            git clone https://github.com/yourusername/reef-server.git /opt/reef_server
            ;;
        [Cc]* )
            log_info "Copying current files to /opt/reef_server..."
            cp -r ./* /opt/reef_server/
            chown -R $SUDO_USER:$SUDO_USER /opt/reef_server
            ;;
        * )
            log_info "Skipping repository setup, using existing files..."
            ;;
    esac
else
    read -p "Would you like to clone the repository? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Creating directory and copying current files..."
        mkdir -p /opt/reef_server
        cp -r ./* /opt/reef_server/
        chown -R $SUDO_USER:$SUDO_USER /opt/reef_server
    else
        log_info "Cloning Reef server..."
        git clone https://github.com/yourusername/reef-server.git /opt/reef_server
    fi
fi

# Add Python 3.9 installation with deadsnakes PPA
install_python39() {
    log_info "Adding deadsnakes PPA for Python 3.9..."
    apt update
    apt install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    
    log_info "Installing Python 3.9..."
    apt update
    apt install -y python3.9 python3.9-venv python3.9-dev
    
    # Verify installation
    if ! command -v python3.9 &> /dev/null; then
        log_error "Failed to install Python 3.9"
        exit 1
    fi
}

# Cleanup function
cleanup_environment() {
    log_info "Cleaning up existing Python environment..."
    
    # Deactivate venv if it's active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    
    # Remove existing venv
    if [ -d "/opt/reef_server/venv" ]; then
        rm -rf /opt/reef_server/venv
        log_info "Removed existing virtual environment"
    fi
}

# Clean up existing environment
cleanup_environment

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null; then
    install_python39
fi

# Setup Python virtual environment with Python 3.9
log_info "Setting up Python environment..."
cd /opt/reef_server
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install all required packages
log_info "Installing main requirements..."
pip install \
    flask \
    pillow \
    requests \
    PyYAML \
    gunicorn \
    psutil \
    opencv-python \
    sdnotify \
    python-dateutil \
    "numpy<2.0.0" \
    werkzeug \
    logging-handler \
    systemd-python

# Install TFLite Runtime and PyCoral
log_info "Installing TFLite Runtime and PyCoral..."
pip uninstall -y tflite-runtime pycoral  # Remove any existing versions

# Install compatible versions
log_info "Installing compatible versions for Python 3.9..."
$VIRTUAL_ENV/bin/pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
$VIRTUAL_ENV/bin/pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# Verify all required packages are installed
log_info "Verifying all required packages..."
required_packages=(
    "flask"
    "PIL"  # for pillow
    "requests"
    "yaml"
    "gunicorn"
    "psutil"
    "cv2"
    "sdnotify"
    "dateutil"
    "werkzeug"
    "numpy"
    "logging"
    "json"
    "time"
    "os"
    "sys"
    "threading"
    "datetime"
    "tflite_runtime"
    "pycoral"
)

for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        log_error "Missing required package: $package"
        exit 1
    fi
done

log_info "All required packages installed successfully"

# Function to check installation with proper Python path
check_installation() {
    $VIRTUAL_ENV/bin/python3 -c "
import sys
print('Python version:', sys.version)
import numpy
print('NumPy version:', numpy.__version__)
import tflite_runtime
import pycoral
from pycoral.utils import edgetpu
print('tflite_runtime version:', tflite_runtime.__version__)
print('pycoral version:', pycoral.__version__)
print('Successfully imported pycoral.utils.edgetpu')
    "
    return $?
}

# Verify installation with explicit error output
if ! check_installation; then
    log_error "Installation verification failed. Debug information:"
    log_info "Python version in venv:"
    $VIRTUAL_ENV/bin/python3 --version
    log_info "Package versions:"
    $VIRTUAL_ENV/bin/pip list | grep -E "numpy|tflite|coral"
    exit 1
fi

log_info "Python environment setup complete"

# Copy service file and reload systemd
log_info "Installing systemd service..."
cp /opt/reef_server/reef.service /etc/systemd/system/

# Stop the service if it's running
if systemctl is-active --quiet reef; then
    log_info "Stopping existing reef service..."
    systemctl stop reef
fi

# Reload systemd configuration
log_info "Reloading systemd configuration..."
systemctl daemon-reload

# Start service with logging
log_info "Starting service..."
systemctl start reef
sleep 2

# Check service status
if ! systemctl is-active --quiet reef; then
    log_error "Service failed to start. Checking logs..."
    journalctl -u reef -n 50
    exit 1
fi

# Setup udev rules
log_info "Setting up udev rules..."
cat > /etc/udev/rules.d/99-reef-usb.rules << EOL
SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", RUN+="/bin/sh -c 'echo -1 > /sys/bus/usb/devices/\$kernel/power/autosuspend'"
EOL

cat > /etc/udev/rules.d/99-reef-pci.rules << EOL
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x1ac1", ATTR{device}=="0x089a", ATTR{power/control}="on"
EOL

# Reload udev rules
udevadm control --reload-rules
udevadm trigger

# Download models using model manager
log_info "Setting up models..."
if [ -f "/opt/reef_server/model_manager.py" ]; then
    cd /opt/reef_server
    chmod +x model_manager.py
    
    # Define available models
    declare -A models
    models[1]="ssd_mobilenet_v2"
    models[2]="efficientdet_lite0"
    models[3]="person_detection"
    
    # Show menu
    log_info "Available pre-trained models:"
    echo "--------------------------------------------------"
    echo "1) ssd_mobilenet_v2"
    echo "   Description: General object detection (COCO dataset)"
    echo "--------------------------------------------------"
    echo "2) efficientdet_lite0"
    echo "   Description: Efficient object detection, good balance of speed/accuracy"
    echo "--------------------------------------------------"
    echo "3) person_detection"
    echo "   Description: Optimized for person detection"
    echo "--------------------------------------------------"
    
    # Get user choice
    while true; do
        read -p "Select model number [1-3] (default: 1): " model_num
        model_num=${model_num:-1}
        
        if [[ "$model_num" =~ ^[1-3]$ ]]; then
            selected_model=${models[$model_num]}
            break
        else
            log_error "Please enter a number between 1 and 3"
        fi
    done
    
    # Download selected model using the number directly
    log_info "Downloading ${selected_model}..."
    ./model_manager.py download $model_num --output-dir /opt/reef_server/models
    
    if [ $? -ne 0 ]; then
        log_error "Failed to download model. Please check the model name and try again."
        exit 1
    fi
else
    log_error "model_manager.py not found in /opt/reef_server/"
    exit 1
fi

# Update service environment variables to use the selected model
log_info "Updating service configuration..."
sed -i "s/REEF_MODEL=.*/REEF_MODEL=${selected_model}.tflite/" /etc/systemd/system/reef.service
sed -i "s/REEF_LABELS=.*/${selected_model}_labels.txt/" /etc/systemd/system/reef.service

# Add this before starting the service
log_info "Verifying installation..."

# Check files exist
files_to_check=(
    "/opt/reef_server/reef_app.py"
    "/opt/reef_server/reef_helpers.py"
    "/opt/reef_server/venv/bin/gunicorn"
    "/etc/systemd/system/reef.service"
    "/var/log/reef"
)

for file in "${files_to_check[@]}"; do
    if [ ! -e "$file" ]; then
        log_error "Missing required file: $file"
        exit 1
    fi
done

# Check Python environment
cd /opt/reef_server
source venv/bin/activate
if ! python3 -c "import flask, pycoral, cv2" 2>/dev/null; then
    log_error "Missing required Python packages"
    exit 1
fi

# Check model files
if [ ! -f "/opt/reef_server/models/${selected_model}.tflite" ]; then
    log_error "Model file not found"
    exit 1
fi

if [ ! -f "/opt/reef_server/models/${selected_model}_labels.txt" ]; then
    log_error "Labels file not found"
    exit 1
fi

# Test log file creation
touch /var/log/reef/reef.log || {
    log_error "Cannot create log file"
    exit 1
}

log_info "Installation complete!"
log_info "API endpoint available at: http://localhost:5000/v1/vision/detection"
log_info "Health check at: http://localhost:5000/health" 
