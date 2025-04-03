WARNING - This is a work in progress, it may not work or require significant tinkering to get to work. 

# Reef Edge TPU Object Detection Server

A production-ready Flask server for real-time object detection using the Edge TPU (USB or PCI/M.2). Following several months of testing the M.2 TPU is where this project is focused, for various reasons the USB TPU is a world of trouble with this application and is going to be out of scope for further development. 
Features automatic device detection, power management, performance monitoring, and robust error handling.

## Prerequisites

### System Packages
```bash
# Update package list
sudo apt update

# Install required system packages
sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    libedgetpu1-std \
    python3-opencv \
    usbutils \
    pciutils \
    curl \
    wget

# If using PCI/M.2 Edge TPU, also install:
sudo apt install -y \
    pcie-tools \
    linux-modules-extra-$(uname -r)
```

### Python Environment Setup
```bash
# Create server directory
sudo mkdir -p /opt/reef_server
sudo chown $USER:$USER /opt/reef_server

# Clone repository
git clone https://github.com/yourusername/reef-server.git /opt/reef_server
cd /opt/reef_server

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install flask pillow requests PyYAML gunicorn psutil opencv-python
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

## Features

- **Multi-Device Support**: 
  - USB Edge TPU
  - PCI/M.2 Edge TPU
  - Automatic device detection and configuration
- **Power Management**: 
  - USB power state management
  - PCI power optimization
  - Prevents device sleep issues
- **Performance Monitoring**: Detailed timing and TPU utilization metrics
- **Error Handling**: Robust error handling with proper HTTP status codes
- **Image Preprocessing**: Automatic image optimization
- **Configurable Timeouts**: Request timeout protection
- **Logging**: Comprehensive logging with rotation

## Hardware Support

### USB Edge TPU
- Google USB Accelerator
- Automatic USB power management
- Hot-plug support
- USB sleep state prevention

### PCI/M.2 Edge TPU
- Google M.2 Accelerator
- PCIe power state optimization
- Native PCIe performance
- Integrated system support

## Installation

### System Requirements:
- Python 3.8+
- Edge TPU (USB or PCI/M.2)
- Linux-based OS (tested on Ubuntu 20.04)

### Device Setup:

#### For USB Edge TPU:
```bash
# Create udev rules file
sudo nano /etc/udev/rules.d/99-reef-usb.rules

# Add this content:
SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", RUN+="/bin/sh -c 'echo -1 > /sys/bus/usb/devices/$kernel/power/autosuspend'"
```

#### For PCI/M.2 Edge TPU:
```bash
# Verify PCI device detection
lspci | grep Google

# Optimize PCI power management
echo 'ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x1ac1", ATTR{device}=="0x089a", ATTR{power/control}="on"' | \
sudo tee /etc/udev/rules.d/99-reef-pci.rules
```

### Apply Rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Server Installation:
```bash
# Create server directory
sudo mkdir -p /opt/reef_server
sudo chown $USER:$USER /opt/reef_server

# Clone repository (if using git)
git clone https://github.com/yourusername/reef-server.git /opt/reef_server
cd /opt/reef_server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install flask pillow requests PyYAML gunicorn psutil
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# Create log directory
sudo mkdir -p /var/log/reef
sudo chown $USER:$USER /var/log/reef
```

### Service Setup:
```bash
# Copy service file
sudo cp reef.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable reef
sudo systemctl start reef
```

## Service Configuration

The service automatically detects and configures the appropriate Edge TPU device:

```bash
# Check device detection
sudo systemctl status reef

# View device type and configuration
sudo journalctl -u reef | grep "Detected"
```

## Monitoring

### Device Status:
```bash
# For USB Edge TPU
sudo cat /sys/bus/usb/devices/*/power/runtime_status

# For PCI Edge TPU
sudo cat /sys/bus/pci/devices/*/power/runtime_status
```

### Performance Monitoring:
```bash
# View real-time logs
tail -f /var/log/reef/reef.log
```

## Troubleshooting

### Device Detection:
```bash
# Check USB devices
lsusb | grep Google

# Check PCI devices
lspci | grep "Global Unichip Corp."
```

### Power Management:
```bash
# USB power settings
find /sys/bus/usb/devices -name power/control -exec cat {} \;

# PCI power settings
find /sys/bus/pci/devices -name power/control -exec cat {} \;
```

### Service Issues:
```bash
sudo systemctl status reef
sudo journalctl -u reef -n 50
```

## Best Practices

1. **Device Selection**:
   - USB Edge TPU: Portable, flexible deployment
   - PCI/M.2 Edge TPU: Better performance, integrated systems

2. **Power Management**:
   - Use provided udev rules
   - Monitor power states
   - Keep firmware updated

3. **Performance Optimization**:
   - Single-threaded operation
   - Direct tensor access
   - Device-specific power settings

## Logging Configuration

The server uses a flexible logging system that can be adjusted based on the environment. By default, the log level is set to `INFO`, but you can switch to `DEBUG` logging for more detailed output.

### Switching to Debug Logging

To enable debug logging, set the `LOG_LEVEL` environment variable to `DEBUG` before starting the server. This will provide more detailed logs, which can be useful for troubleshooting and development.

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Start the server
python3 reef_app.py
```

To revert to the default `INFO` logging level, unset the `LOG_LEVEL` environment variable or set it to `INFO`:

```bash
# Revert to INFO logging
unset LOG_LEVEL
# or
export LOG_LEVEL=INFO
```

## Environment Variables

The server can be configured using the following environment variables:

- `REEF_MODELS_DIR`: Directory containing model files (default: `/opt/reef_server/models`)
- `REEF_MODEL`: Name of the model file (default: `person_detection.tflite`)
- `REEF_LABELS`: Name of the labels file (default: `person_detection_labels.txt`)
- `LOG_LEVEL`: Set the logging level (`DEBUG`, `INFO`, etc.)

## API Endpoints

### POST /predict
Performs object detection on an uploaded image.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response**:
```json
{
    "predictions": [
        {
            "bbox": [x1, y1, x2, y2],
            "class": "label",
            "score": 0.95
        }
    ],
    "inference_time": 15.5
}
```

## Benchmarking

The server includes a comprehensive benchmarking tool to test performance and reliability. 

### Requirements
```bash
# Install benchmark dependencies
pip install tqdm aiohttp pandas numpy matplotlib seaborn
```

### Usage
```bash
python benchmark.py [options]
```

### Options
- `--url`: Server URL (default: http://localhost:5000)
- `--iterations`: Number of test iterations (default: 100)
- `--concurrent`: Number of concurrent requests (default: 1)
- `--output`: Output directory for results (default: benchmark_results)
- `--mode`: Benchmark mode (choices: quick, normal, stress, endurance)
  - quick: 10 iterations
  - normal: 100 iterations
  - stress: 1000 iterations, 10 concurrent requests
  - endurance: 10000 iterations, 2 concurrent requests
- `--delay`: Delay between requests in seconds (default: 0)

### Example Commands
```bash
# Quick test
python benchmark.py --mode quick

# Stress test
python benchmark.py --mode stress --url http://localhost:5000

# Custom test
python benchmark.py --iterations 500 --concurrent 5 --delay 0.1
```

### Output
The benchmark tool provides:
- Request success/failure rates
- Timing statistics (mean, median, percentiles)
- System resource usage
- Prediction statistics
- Detailed CSV reports for further analysis

Results are saved to:
- `benchmark_results_TIMESTAMP.csv`: Detailed test results
- `system_metrics_TIMESTAMP.csv`: System resource metrics
