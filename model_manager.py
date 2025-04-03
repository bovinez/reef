#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import requests
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common pre-trained models
PRETRAINED_MODELS = {
    "ssd_mobilenet_v2": {
        "model_url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v2_1.0_quant_2018_06_29.zip",
        "label_url": "https://dl.google.com/coral/canned_models/coco_labels.txt",
        "description": "General object detection (COCO dataset)",
    },
    "efficientdet_lite0": {
        "model_url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/efficientdet_lite0_edgetpu_1.0_quant.tflite",
        "label_url": "https://dl.google.com/coral/canned_models/coco_labels.txt",
        "description": "Efficient object detection, good balance of speed/accuracy",
    },
    "person_detection": {
        "model_url": "https://github.com/google-coral/test_data/raw/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
        "label_url": "https://dl.google.com/coral/canned_models/coco_labels.txt",
        "description": "Optimized for person detection",
    }
}

def download_file(url, destination):
    """Download a file from URL to destination."""
    logger.info(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def convert_to_edge_tpu(model_path):
    """Convert TFLite model to Edge TPU model."""
    output_path = str(model_path).replace('.tflite', '_edgetpu.tflite')
    logger.info(f"Converting model to Edge TPU format: {output_path}")
    
    try:
        subprocess.run([
            'edgetpu_compiler',
            '--out_dir', os.path.dirname(model_path),
            model_path
        ], check=True)
        logger.info("Model conversion successful")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting model: {e}")
        return None

def download_pretrained_model(model_name, output_dir):
    """Download and prepare a pre-trained model."""
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = PRETRAINED_MODELS[model_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model
    model_path = output_dir / f"{model_name}.tflite"
    download_file(model_info['model_url'], model_path)
    
    # Download labels
    label_path = output_dir / f"{model_name}_labels.txt"
    download_file(model_info['label_url'], label_path)
    
    # Convert to Edge TPU if needed
    if not model_info['model_url'].endswith('_edgetpu.tflite'):
        convert_to_edge_tpu(model_path)
    
    return model_path, label_path

def convert_custom_model(model_path, output_dir, dataset_config=None):
    """Convert a custom model for Edge TPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If dataset config is provided, save it
    if dataset_config:
        config_path = output_dir / 'dataset_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
    
    # Convert model
    edge_tpu_model = convert_to_edge_tpu(model_path)
    if edge_tpu_model:
        logger.info(f"Custom model converted successfully: {edge_tpu_model}")
    return edge_tpu_model

def list_available_models():
    """List all available pre-trained models with numbered options."""
    print("\nAvailable pre-trained models:")
    print("-" * 50)
    
    # Create a numbered list of models
    models_list = list(PRETRAINED_MODELS.items())
    for idx, (name, info) in enumerate(models_list, 1):
        print(f"\n{idx}) {name}:")
        print(f"Description: {info['description']}")
        print("-" * 50)
    
    return models_list

def main():
    parser = argparse.ArgumentParser(description='Coral Edge TPU Model Manager')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List available models
    list_parser = subparsers.add_parser('list', help='List available pre-trained models')
    
    # Download pre-trained model
    download_parser = subparsers.add_parser('download', help='Download a pre-trained model')
    download_parser.add_argument('model_number', type=int, help='Number of the pre-trained model')
    download_parser.add_argument('--output-dir', default='models', help='Output directory')
    
    # Convert custom model
    convert_parser = subparsers.add_parser('convert', help='Convert a custom model')
    convert_parser.add_argument('model_path', help='Path to the custom TFLite model')
    convert_parser.add_argument('--output-dir', default='models', help='Output directory')
    convert_parser.add_argument('--config', help='Path to dataset configuration YAML')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_available_models()
    
    elif args.command == 'download':
        try:
            models_list = list(PRETRAINED_MODELS.items())
            if args.model_number < 1 or args.model_number > len(models_list):
                logger.error(f"Invalid model number. Please choose between 1 and {len(models_list)}")
                sys.exit(1)
                
            model_name = models_list[args.model_number - 1][0]
            model_path, label_path = download_pretrained_model(model_name, args.output_dir)
            logger.info(f"Downloaded model: {model_path}")
            logger.info(f"Downloaded labels: {label_path}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            sys.exit(1)
    
    elif args.command == 'convert':
        config = None
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        
        try:
            converted_model = convert_custom_model(args.model_path, args.output_dir, config)
            if not converted_model:
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 
