import os
import logging
import time
import numpy as np
import cv2
from logging.handlers import RotatingFileHandler
import psutil
from tflite_runtime.interpreter import Interpreter
from pycoral.utils import edgetpu
import threading
import gc

logger = logging.getLogger(__name__)

class EdgeTPUInference:
    def __init__(self, model_path, device_type="usb"):
        if not model_path:
            raise ValueError("Model path must be specified")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.device_type = device_type
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.last_reset_time = 0
        self.performance_stats = {
            'count': 0,
            'total_time': 0,
            'max_time': 0,
            'detection_count': 0
        }
        
        # PCI TPU specific optimizations
        self.is_pci = (device_type == "pci")
        
        logger.info(f"Initializing Edge TPU in {device_type} mode")
        
        # Initialize the TPU
        self._initialize_tpu()
        
    def _initialize_tpu(self):
        """Initialize the TPU with timeout protection"""
        try:
            # If we already have an interpreter, clean it up
            if self.interpreter is not None:
                self.interpreter = None
                # Force garbage collection
                gc.collect()
                
            logger.info(f"Initializing TPU with model: {self.model_path}")
            
            # PCI initialization has different parameters
            if self.is_pci:
                logger.info("Using PCI EdgeTPU initialization")
                try:
                    # For PCI Edge TPU, try to be more specific about the device
                    self.interpreter = edgetpu.make_interpreter(
                        str(self.model_path),
                        device=':0'  # First PCI device
                    )
                    logger.info("Successfully initialized PCI Edge TPU with device :0")
                except Exception as pci_e:
                    logger.warning(f"Failed to initialize with device :0 - {pci_e}")
                    # Fallback to default initialization
                    self.interpreter = edgetpu.make_interpreter(str(self.model_path))
                    logger.info("Fallback: initialized PCI Edge TPU with default settings")
            else:
                # USB initialization
                self.interpreter = edgetpu.make_interpreter(str(self.model_path))
                
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3]
            
            # Log model details for debugging
            logger.info(f"Model loaded: {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output details: {len(self.output_details)} tensors")
            
            # Perform a warmup inference
            self._warmup()
            
            # Reset error count after successful initialization
            self.consecutive_errors = 0
            self.last_reset_time = time.time()
            
            # Reset performance stats
            self.performance_stats = {
                'count': 0,
                'total_time': 0,
                'max_time': 0,
                'detection_count': 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TPU: {str(e)}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.critical(f"Failed to initialize TPU after {self.consecutive_errors} attempts")
                
            raise

    def _warmup(self):
        """Perform a warm-up inference to initialize the TPU"""
        try:
            logger.info("Performing warm-up inference")
            # Create a dummy image matching the input shape
            dummy_input = np.zeros(
                (1, self.input_shape[0], self.input_shape[1], 3), 
                dtype=np.uint8
            )
            
            # Set a timeout for the warmup
            start_time = time.time()
            max_time = 5  # 5 second timeout for warmup
            
            # Set tensor and invoke with timeout protection
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                dummy_input
            )
            
            # For PCI TPU, we do two warmup passes for better performance
            warmup_passes = 2 if self.is_pci else 1
            
            for i in range(warmup_passes):
                # Run warmup in a separate thread to allow timeout
                def run_invoke():
                    try:
                        self.interpreter.invoke()
                        return True
                    except Exception as e:
                        logger.error(f"Warmup invoke failed: {e}")
                        return False
                
                # Run in thread
                invoke_thread = threading.Thread(target=run_invoke)
                invoke_thread.daemon = True
                invoke_thread.start()
                
                # Wait with timeout
                invoke_thread.join(timeout=max_time)
                
                if invoke_thread.is_alive():
                    logger.error(f"Warmup pass {i+1} timed out after {max_time} seconds!")
                    if i == 0:  # Only continue to the next pass if first one times out
                        continue
                    else:
                        return False
            
            logger.info(f"Warm-up inference completed successfully with {warmup_passes} passes")
            return True
        except Exception as e:
            logger.error(f"Warm-up inference failed: {e}")
            return False

    def predict(self, image, threshold=0.4):
        """Run inference with better error handling and recovery"""
        try:
            # Check if we need to re-initialize
            if self.interpreter is None or self.input_details is None:
                logger.warning("TPU interpreter not initialized, attempting to initialize")
                self._initialize_tpu()
                
            total_start = time.time()
            
            # Prepare input
            preprocess_start = time.time()
            resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # Set input tensor directly
            try:
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    input_data
                )
            except Exception as e:
                logger.error(f"Failed to set input tensor: {e}")
                # Attempt recovery
                self._initialize_tpu()
                # Retry once
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    input_data
                )
            
            # Run inference and time it
            tpu_start = time.time()
            
            try:
                self.interpreter.invoke()
            except Exception as e:
                logger.error(f"Invoke failed: {e}")
                # Try to reinitialize and retry once
                if time.time() - self.last_reset_time > 30:  # Limit resets to once per 30 seconds
                    logger.warning("Attempting TPU reset due to invoke failure")
                    self._initialize_tpu()
                    # Set tensor again after reset
                    self.interpreter.set_tensor(
                        self.input_details[0]['index'],
                        input_data
                    )
                    # Retry invoke
                    self.interpreter.invoke()
                else:
                    raise RuntimeError(f"TPU invoke failed and reset recently attempted: {e}")
                    
            tpu_time = (time.time() - tpu_start) * 1000
            
            # Get output data with proper copying and error handling
            postprocess_start = time.time()
            
            try:
                # Create explicit copies of the output tensors to avoid keeping references to TPU memory
                boxes = np.copy(self.interpreter.get_tensor(self.output_details[0]['index'])[0])
                classes = np.copy(self.interpreter.get_tensor(self.output_details[1]['index'])[0])
                scores = np.copy(self.interpreter.get_tensor(self.output_details[2]['index'])[0])
            except Exception as e:
                logger.error(f"Failed to get output tensors: {e}")
                raise RuntimeError(f"TPU output processing failed: {e}")
            
            # Log inference time if it's unusually long
            if tpu_time > 5000:  # 5 seconds
                logger.warning(f"Unusually long inference time: {tpu_time:.2f}ms")
            
            # Delete input data to free memory early
            del input_data
            del resized
            
            # For PCIe TPU, use parallel processing for filtering if many detections
            total_detections = len(scores)
            detection_threshold = 500 if self.is_pci else 100  # Higher threshold for PCI TPU
            
            # Filter results - different approach for large numbers of detections
            if total_detections > detection_threshold and self.is_pci:
                # For large numbers of detections on PCI TPU, process in chunks
                logger.debug(f"Using optimized filtering for {total_detections} detections")
                
                # Create valid mask first (faster than direct indexing for large arrays)
                valid_mask = scores > threshold
                valid_count = np.sum(valid_mask)
                
                # Only if we have valid detections, extract them
                if valid_count > 0:
                    filtered_boxes = boxes[valid_mask]
                    filtered_classes = classes[valid_mask]
                    filtered_scores = scores[valid_mask]
                else:
                    # No valid detections
                    filtered_boxes = np.array([], dtype=boxes.dtype).reshape(0, 4)
                    filtered_classes = np.array([], dtype=classes.dtype)
                    filtered_scores = np.array([], dtype=scores.dtype)
            else:
                # Standard approach for smaller numbers
                valid_indices = scores > threshold
                filtered_boxes = np.copy(boxes[valid_indices])
                filtered_classes = np.copy(classes[valid_indices])
                filtered_scores = np.copy(scores[valid_indices])
            
            # Clean up original arrays
            del boxes
            del classes
            del scores
            
            results = (
                filtered_boxes,
                filtered_classes,
                filtered_scores
            )
            
            postprocess_time = (time.time() - postprocess_start) * 1000
            total_time = (time.time() - total_start) * 1000
            
            # Update performance stats
            self.performance_stats['count'] += 1
            self.performance_stats['total_time'] += total_time
            self.performance_stats['max_time'] = max(self.performance_stats['max_time'], total_time)
            self.performance_stats['detection_count'] += len(filtered_scores)
            
            # If too many detections, log it
            if len(filtered_scores) > 100:
                logger.warning(f"Large number of detections: {len(filtered_scores)}")
            
            # Reset error counter on success
            self.consecutive_errors = 0
            
            # Force garbage collection to free up memory
            if len(filtered_scores) > detection_threshold:
                gc.collect()
            
            # Log detailed performance
            self.log_performance({
                'total': total_time,
                'preprocess': preprocess_time,
                'inference': tpu_time,
                'postprocess': postprocess_time,
                'detections': len(filtered_scores)
            })
            
            return results
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error during inference (attempt {self.consecutive_errors}): {str(e)}", exc_info=True)
            
            # If we've had multiple consecutive errors, try to reset the TPU
            if self.consecutive_errors >= 2 and time.time() - self.last_reset_time > 60:
                logger.warning(f"Multiple inference errors, attempting TPU reset")
                try:
                    self._initialize_tpu()
                except Exception as reset_error:
                    logger.error(f"TPU reset failed: {reset_error}")
            
            # Try to recover by forcing garbage collection
            gc.collect()
            raise

    def log_performance(self, perf_data):
        """Log performance metrics with detection count included"""
        detection_count = perf_data.get('detections', 0)
        
        # Log at info level for large detection counts, debug for normal operation
        if detection_count > 50:
            logger.info(f"Performance metrics for {detection_count} detections:")
            logger.info(f"├── Total: {perf_data['total']:.2f}ms")
            logger.info(f"├── Preprocess: {perf_data['preprocess']:.2f}ms")
            logger.info(f"├── Inference: {perf_data['inference']:.2f}ms")
            logger.info(f"├── Postprocess: {perf_data['postprocess']:.2f}ms")
            
            # For PCI TPU with many detections, log average performance
            if self.is_pci and self.performance_stats['count'] > 10 and self.performance_stats['count'] % 10 == 0:
                avg_time = self.performance_stats['total_time'] / self.performance_stats['count']
                avg_detections = self.performance_stats['detection_count'] / self.performance_stats['count']
                logger.info(f"PCI TPU Performance Summary (last {self.performance_stats['count']} requests):")
                logger.info(f"├── Average time: {avg_time:.2f}ms")
                logger.info(f"├── Max time: {self.performance_stats['max_time']:.2f}ms")
                logger.info(f"├── Average detections: {avg_detections:.1f}")
        else:
            logger.debug(f"Performance metrics for {detection_count} detections:")
            logger.debug(f"├── Total: {perf_data['total']:.2f}ms")
            logger.debug(f"├── Preprocess: {perf_data['preprocess']:.2f}ms")
            logger.debug(f"├── Inference: {perf_data['inference']:.2f}ms")
            logger.debug(f"├── Postprocess: {perf_data['postprocess']:.2f}ms")

    def get_performance_stats(self):
        """Return performance statistics"""
        if self.performance_stats['count'] == 0:
            return {
                'count': 0,
                'avg_time': 0,
                'max_time': 0,
                'avg_detections': 0
            }
            
        return {
            'count': self.performance_stats['count'],
            'avg_time': self.performance_stats['total_time'] / self.performance_stats['count'],
            'max_time': self.performance_stats['max_time'],
            'avg_detections': self.performance_stats['detection_count'] / self.performance_stats['count']
        }

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start(self):
        self.start_time = time.time()
        self.checkpoints = {}

    def checkpoint(self, name):
        if self.start_time is None:
            raise RuntimeError("Monitor not started")
        self.checkpoints[name] = time.time() - self.start_time

    def log_performance(self):
        if not self.checkpoints:
            return ""
        total_time = max(self.checkpoints.values()) * 1000  # Convert to ms
        lines = [f"Performance:"]
        lines.append(f"├── Total: {total_time:.2f}ms")
        sorted_checkpoints = sorted(self.checkpoints.items(), key=lambda x: x[1])
        for name, timestamp in sorted_checkpoints:
            lines.append(f"├── {name}: {timestamp*1000:.2f}ms")
        return "\n".join(lines)

class MonitoredRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_rollover = time.time()
        self.log_count = 0
        self.CHECK_INTERVAL = 300  # 5 minutes
        self._logging_status = True
        self._last_status_print = 0
        self.STATUS_PRINT_INTERVAL = 60  # Print status max once per minute

    def emit(self, record):
        """Emit a record with rate monitoring"""
        try:
            # Always emit the actual log
            super().emit(record)
            
            # Update counters
            current_time = time.time()
            self.log_count += 1
            
            # Check if we should print status (but avoid recursive logging)
            if (current_time - self.last_rollover >= self.CHECK_INTERVAL and 
                current_time - self._last_status_print >= self.STATUS_PRINT_INTERVAL):
                
                # Use print instead of logging to avoid recursion
                print(f"[Logging Monitor] {self.log_count} logs in last {self.CHECK_INTERVAL} seconds")
                
                # Reset counters
                self.log_count = 0
                self.last_rollover = current_time
                self._last_status_print = current_time
                
        except Exception as e:
            # Use sys.stderr for errors to avoid recursion
            import sys
            print(f"Error in log handler: {str(e)}", file=sys.stderr)

class HealthMonitor:
    def __init__(self, coral_engine=None, logger=None):
        self._is_healthy = True
        self.last_check = time.time()
        self.check_interval = 60  # seconds
        self.coral_engine = coral_engine
        self.logger = logger or logging.getLogger(__name__)
        self._start_monitoring()

    def check_health(self):
        """Method to check health status"""
        # Add more detailed health checks
        try:
            process = psutil.Process()
            
            # Check memory usage - if over 90%, consider unhealthy
            if process.memory_percent() > 90:
                self.logger.warning("High memory usage detected: {:.1f}%".format(process.memory_percent()))
                return False
                
            # Check CPU usage - if sustained over 95%, consider unhealthy
            if process.cpu_percent(interval=0.5) > 95:
                self.logger.warning("High CPU usage detected")
                return False
                
            # Check if we have too many open file descriptors
            if hasattr(process, 'num_fds') and process.num_fds() > 950:  # Approaching default 1024 limit
                self.logger.warning(f"Too many open file descriptors: {process.num_fds()}")
                return False
                
            return self._is_healthy
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return False

    @property
    def is_healthy(self):
        """Property to get health status"""
        return self._is_healthy

    def _start_monitoring(self):
        def monitor():
            while True:
                try:
                    # Health check logic here
                    if self.coral_engine:
                        # Try to verify TPU is responsive
                        if hasattr(self.coral_engine, '_warmup'):
                            # Instead of calling warmup each time (which is expensive),
                            # just check that the interpreter exists and attributes are accessible
                            if (not hasattr(self.coral_engine, 'interpreter') or 
                                self.coral_engine.interpreter is None or
                                not hasattr(self.coral_engine, 'input_details')):
                                self.logger.error("TPU interpreter not accessible")
                                self._is_healthy = False
                            else:
                                self._is_healthy = True
                                
                    # Check system resources
                    process = psutil.Process()
                    memory_percent = process.memory_percent()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    # Log resource usage periodically
                    self.logger.debug(f"Health monitor - Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%")
                    
                    # If memory is critically high, try to recover
                    if memory_percent > 85:
                        self.logger.warning(f"Memory usage high ({memory_percent:.1f}%), triggering garbage collection")
                        gc.collect()
                    
                    # Check for too many open connections
                    try:
                        open_connections = len(process.connections())
                        if open_connections > 100:  # Arbitrary limit, adjust as needed
                            self.logger.warning(f"High number of open connections: {open_connections}")
                    except Exception as conn_e:
                        self.logger.error(f"Error checking connections: {conn_e}")
                    
                    self.last_check = time.time()
                    time.sleep(self.check_interval)
                except Exception as e:
                    self._is_healthy = False
                    self.logger.error(f"Health check failed: {e}")

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def get_status(self):
        """Get detailed status information"""
        process = psutil.Process()
        return {
            "healthy": self._is_healthy,
            "last_check": self.last_check,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }

class RequestMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_report = self.start_time
        self.REPORT_INTERVAL = 3600  # 1 hour
        
    def check_resources(self):
        """Check system resources"""
        process = psutil.Process()
        return {
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }

    def log_request(self, success=True):
        self.request_count += 1
        if not success:
            self.error_count += 1
            
        current_time = time.time()
        if current_time - self.last_report >= self.REPORT_INTERVAL:
            uptime = current_time - self.start_time
            success_rate = ((self.request_count - self.error_count) / 
                          max(self.request_count, 1) * 100)
            
            logger.info(f"Server uptime: {uptime/3600:.1f} hours")
            logger.info(f"Total requests: {self.request_count}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            self.last_report = current_time 

    def log_inference_results(self, results):
        for result in results:
            logger.debug(f"Found {result['label']} with confidence {result['confidence']:.2f}")
