import os
import argparse
import sys
import logging
import flask
import cv2
import numpy as np
from functools import wraps
import time
import threading
from reef_helpers import EdgeTPUInference, PerformanceMonitor, MonitoredRotatingFileHandler, HealthMonitor, RequestMonitor
from logging.handlers import RotatingFileHandler
import json
import resource
import psutil
import signal
import atexit
import sdnotify
from concurrent.futures import ThreadPoolExecutor
import subprocess
import logging.handlers
import queue
import traceback
from werkzeug.exceptions import ServiceUnavailable

# Set higher soft limit for file descriptors
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Create a queue for log messages
log_queue = queue.Queue(-1)

# Create a handler that writes log messages to a file
file_handler = logging.handlers.RotatingFileHandler(
    '/var/log/reef/reef-server.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
file_handler.setFormatter(formatter)

# Create a QueueHandler and set it to use the queue
queue_handler = logging.handlers.QueueHandler(log_queue)

# Set up the root logger to use the QueueHandler
logging.basicConfig(level=logging.INFO, handlers=[queue_handler])

# Create a QueueListener to listen for log messages and write them to the file
listener = logging.handlers.QueueListener(log_queue, file_handler)
listener.start()

# Create a logger for your application
logger = logging.getLogger(__name__)

def handle_logging_exception(exc_type, exc_value, exc_traceback):
    """Handle any logging exceptions"""
    try:
        # Try to log the error
        logger.error("Logging error occurred", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Attempt to reset handlers
        for handler in logger.handlers:
            try:
                handler.close()
            except:
                pass
            
        # Reinitialize logging
        logger.handlers = []
        logger.addHandler(queue_handler)
        
    except:
        # If all else fails, print to stderr
        import sys
        print("Logging failed completely:", file=sys.stderr)
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)

# Global flag for graceful shutdown
is_shutting_down = False

# At the top with other globals
coral_engine = None
health_monitor = None

# Global lock for TPU access with timeout tracking
tpu_lock = threading.Lock()
tpu_lock_owner = None
tpu_lock_acquired_time = 0
tpu_lock_state_lock = threading.Lock()  # Meta-lock to protect the lock state variables

# TPU diagnostic counters
tpu_stats = {
    'inference_count': 0,
    'successful_inferences': 0,
    'failed_inferences': 0,
    'reset_count': 0,
    'last_reset_time': 0,
    'longest_inference_time': 0,
    'inference_times': [],  # Store recent inference times (last 100)
    'consecutive_errors': 0,
    'last_error': None,
    'last_error_time': 0,
    'deadlocks_detected': 0
}
tpu_stats_lock = threading.Lock()

# Enhanced concurrency for throughput
MAX_CONCURRENT_REQUESTS = 64  # Increased from 16 to 64
request_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

# Request tracking (for diagnostics only, not for blocking)
active_requests = 0
active_requests_lock = threading.Lock()
request_history = []  # Track timing of recent requests
request_history_lock = threading.Lock()
MAX_HISTORY_SIZE = 200  # Keep history of last 200 requests

# Request watchdog data
active_request_threads = {}  # Maps thread_id -> {'start_time': time, 'operation': str}
active_request_threads_lock = threading.Lock()
# Maximum time allowed for a single request before we log a warning
MAX_REQUEST_TIME = 60  # seconds

# TPU diagnostics timeout - for logging only, not for blocking
TPU_DIAGNOSTICS_THRESHOLD = 30  # seconds - log when TPU operations exceed this
TPU_OPERATION_TIMEOUT = 20  # Hard timeout for TPU operations
tpu_last_success = time.time()
tpu_timeout_lock = threading.Lock()
# Track API activity separately from TPU activity
last_api_request_time = time.time()
api_request_lock = threading.Lock()

# Thread dump settings
enable_thread_dumps = True
thread_dump_interval = 60  # seconds between dumps when issues detected
last_thread_dump_time = 0

# Heartbeat for monitoring
last_heartbeat = time.time()
heartbeat_lock = threading.Lock()
expected_heartbeat_interval = 10  # seconds

# Add these near the top of the file with other global variables
# Request tracking variables
_request_stats_lock = threading.Lock()
_request_counter = 0
_total_requests = 0
_successful_requests = 0
_active_requests = set()
_request_start_times = {}
_request_durations = []

def update_heartbeat():
    """Update the heartbeat timestamp"""
    global last_heartbeat
    with heartbeat_lock:
        last_heartbeat = time.time()

def check_heartbeat():
    """Check if heartbeat is current, indicating system responsiveness"""
    with heartbeat_lock:
        return time.time() - last_heartbeat < expected_heartbeat_interval * 2

def dump_all_threads():
    """Dump all thread stacks for debugging deadlocks"""
    global last_thread_dump_time
    
    current_time = time.time()
    # Limit frequency of thread dumps to avoid log flooding
    if not enable_thread_dumps or current_time - last_thread_dump_time < thread_dump_interval:
        return
    
    last_thread_dump_time = current_time
    
    logger.critical("=== THREAD DUMP START ===")
    for thread_id, thread in threading._active.items():
        if thread:
            stack = traceback.format_stack(sys._current_frames()[thread_id])
            logger.critical(f"Thread {thread.name} ({thread_id}):")
            for line in stack:
                logger.critical(line.strip())
    logger.critical("=== THREAD DUMP END ===")

def monitored_lock_acquire(lock, timeout=10, operation_name="unknown", collect_stack=True):
    """Wrapper for lock acquisition with monitoring and timeout"""
    global tpu_lock_owner, tpu_lock_acquired_time
    
    if collect_stack:
        stack = ''.join(traceback.format_stack())
    else:
        stack = "Stack collection disabled"
    
    thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    
    # Try to acquire the lock with timeout
    start_time = time.time()
    acquired = lock.acquire(timeout=timeout)
    
    if acquired:
        # Record lock ownership
        with tpu_lock_state_lock:
            tpu_lock_owner = f"{thread_name} ({thread_id})"
            tpu_lock_acquired_time = time.time()
            
        logger.debug(f"Thread {thread_name} ({thread_id}) acquired {operation_name} lock after {time.time() - start_time:.2f}s")
        
        # Register active operation
        with active_request_threads_lock:
            active_request_threads[thread_id] = {
                'operation': operation_name,
                'start_time': time.time(),
                'stack': stack
            }
        
        return True
    else:
        logger.error(f"Thread {thread_name} ({thread_id}) timed out acquiring {operation_name} lock after {timeout}s")
        logger.error(f"Current lock owner: {tpu_lock_owner}, held for {time.time() - tpu_lock_acquired_time:.2f}s")
        
        # Dump thread stacks to diagnose the deadlock
        dump_all_threads()
        
        with tpu_stats_lock:
            tpu_stats['deadlocks_detected'] += 1
        
        return False

def monitored_lock_release(lock, operation_name="unknown"):
    """Wrapper for lock release with monitoring"""
    global tpu_lock_owner, tpu_lock_acquired_time
    
    thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    
    # Clear lock ownership
    with tpu_lock_state_lock:
        tpu_lock_owner = None
        tpu_lock_acquired_time = 0
    
    # Remove from active operations
    with active_request_threads_lock:
        if thread_id in active_request_threads:
            del active_request_threads[thread_id]
    
    # Release the lock
    lock.release()
    logger.debug(f"Thread {thread_name} ({thread_id}) released {operation_name} lock")

def cleanup():
    """Cleanup function to be called on shutdown"""
    global is_shutting_down, coral_engine
    is_shutting_down = True
    logger.info("Cleaning up resources...")
    try:
        if coral_engine is not None:
            del coral_engine
        logging.shutdown()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global is_shutting_down, coral_engine
    
    logger.info(f"Received signal {signum}")
    logger.info("Cleaning up resources...")
    is_shutting_down = True
    
    try:
        if coral_engine:
            coral_engine.interpreter = None
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Initialize Flask app
app = flask.Flask(__name__)

# Get settings from environment variables with fallbacks
models_dir = os.getenv('REEF_MODELS_DIR', '/opt/reef_server/models')
model_file = os.getenv('REEF_MODEL', 'person_detection.tflite')
labels_file = os.getenv('REEF_LABELS', 'person_detection_labels.txt')

# Construct full paths
MODEL = os.path.abspath(os.path.join(models_dir, model_file))
LABELS = os.path.abspath(os.path.join(models_dir, labels_file))

def load_labels(path):
    """Load labels with their correct class IDs"""
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                # Split on whitespace and expect "id label"
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    label = parts[1]
                    labels[class_id] = label
    return labels

def log_timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start
        if duration > 10:
            logging.warning(f"Request took {duration:.2f} seconds")
        return result
    return wrapper

def update_tpu_success():
    """Update the timestamp of the last successful TPU operation"""
    global tpu_last_success
    with tpu_timeout_lock:
        tpu_last_success = time.time()

def log_tpu_stats():
    """Log current TPU statistics (without acquiring locks if already held)"""
    # Make a copy of the stats
    with tpu_stats_lock:
        stats = dict(tpu_stats)  # Make a copy to avoid race conditions
    
    # Process and log stats outside the lock
    # Calculate average inference time from recent history
    avg_time = 0
    if stats['inference_times']:
        avg_time = sum(stats['inference_times']) / len(stats['inference_times'])
    
    success_rate = 0
    if stats['inference_count'] > 0:
        success_rate = (stats['successful_inferences'] / stats['inference_count']) * 100
        
    logger.info(f"TPU Stats: {stats['inference_count']} inferences, "
                f"{success_rate:.1f}% success rate, "
                f"{stats['reset_count']} resets, "
                f"{stats['deadlocks_detected']} deadlocks, "
                f"avg time: {avg_time:.2f}s, "
                f"max time: {stats['longest_inference_time']:.2f}s")

def update_tpu_stats(success=True, inference_time=None, error=None):
    """Update TPU statistics"""
    # Get a local copy of all values we need to check/modify before acquiring the lock
    # This minimizes the time spent holding the lock
    with tpu_stats_lock:
        tpu_stats['inference_count'] += 1
        
        if success:
            tpu_stats['successful_inferences'] += 1
            tpu_stats['consecutive_errors'] = 0
        else:
            tpu_stats['failed_inferences'] += 1
            tpu_stats['consecutive_errors'] += 1
            tpu_stats['last_error'] = str(error)
            tpu_stats['last_error_time'] = time.time()
        
        if inference_time is not None:
            # Keep only the last 100 inference times
            tpu_stats['inference_times'].append(inference_time)
            if len(tpu_stats['inference_times']) > 100:
                tpu_stats['inference_times'].pop(0)
            
            # Update longest inference time
            tpu_stats['longest_inference_time'] = max(
                tpu_stats['longest_inference_time'], 
                inference_time
            )
        
        # Calculate if we need to log stats
        should_log = (tpu_stats['inference_count'] % 100 == 0)
        
        # If we need to log stats, make a copy of the stats while holding the lock
        stats_copy = None
        if should_log:
            stats_copy = dict(tpu_stats)
    
    # If needed, log stats OUTSIDE of the lock to avoid potential deadlocks
    if should_log and stats_copy:
        # Calculate average inference time from recent history
        avg_time = 0
        if stats_copy['inference_times']:
            avg_time = sum(stats_copy['inference_times']) / len(stats_copy['inference_times'])
        
        success_rate = 0
        if stats_copy['inference_count'] > 0:
            success_rate = (stats_copy['successful_inferences'] / stats_copy['inference_count']) * 100
            
        logger.info(f"TPU Stats: {stats_copy['inference_count']} inferences, "
                    f"{success_rate:.1f}% success rate, "
                    f"{stats_copy['reset_count']} resets, "
                    f"{stats_copy['deadlocks_detected']} deadlocks, "
                    f"avg time: {avg_time:.2f}s, "
                    f"max time: {stats_copy['longest_inference_time']:.2f}s")

def track_api_activity():
    """Update the timestamp of the last API request"""
    global last_api_request_time
    with api_request_lock:
        last_api_request_time = time.time()

def check_tpu_issues():
    """Check if the TPU has been unresponsive for too long (for diagnostics only)"""
    with tpu_timeout_lock:
        last_success_age = time.time() - tpu_last_success
        
        # Only log warnings if we've had API activity recently
        with api_request_lock:
            time_since_last_request = time.time() - last_api_request_time
            
            # Only warn if: 
            # 1. The TPU hasn't had a success in a while (exceeding threshold)
            # 2. There has been recent API activity (within threshold)
            # 3. There's a significant gap between the last API request and last TPU success
            if (last_success_age > TPU_DIAGNOSTICS_THRESHOLD and 
                time_since_last_request < TPU_DIAGNOSTICS_THRESHOLD and
                last_success_age - time_since_last_request > 10):  # At least 10s gap
                
                logger.warning(f"TPU operation delay detected! Last success: {last_success_age:.1f}s ago, "
                              f"Last API request: {time_since_last_request:.1f}s ago, "
                              f"Gap: {last_success_age - time_since_last_request:.1f}s")
                return True
    return False

def monitor_active_requests():
    """Monitor active requests for stuck operations"""
    with active_request_threads_lock:
        current_time = time.time()
        for thread_id, info in list(active_request_threads.items()):
            duration = current_time - info['start_time']
            if duration > MAX_REQUEST_TIME:
                logger.error(f"Stuck operation detected! Thread {thread_id} in '{info['operation']}' "
                            f"for {duration:.1f}s")
                
                # Only dump stack for the first detection to avoid log flooding
                if duration < MAX_REQUEST_TIME + 10:
                    logger.error(f"Stack trace for stuck thread {thread_id}:\n{info['stack']}")
                    dump_all_threads()

def reset_tpu_if_needed():
    """Reset the TPU if consecutive errors exceed threshold or timeout is detected"""
    global coral_engine
    
    # Only reset if we have consecutive errors or timeout is detected
    with tpu_stats_lock:
        should_reset = (
            tpu_stats['consecutive_errors'] >= 3 or
            (time.time() - tpu_last_success > TPU_DIAGNOSTICS_THRESHOLD and
             time.time() - tpu_stats['last_reset_time'] > 60)  # Limit resets to once per minute
        )
        
        if not should_reset:
            return False
    
    logger.warning("Attempting to reset TPU due to consecutive errors or timeout")
    try:
        # Attempt to reinitialize the TPU
        acquired = monitored_lock_acquire(tpu_lock, timeout=30, operation_name="TPU_RESET")
        if not acquired:
            logger.error("Failed to acquire lock for TPU reset - possible deadlock")
            return False
            
        try:
            device_type = coral_engine.device_type
            model_path = coral_engine.model_path
            
            # Explicitly delete and reconstruct
            del coral_engine
            coral_engine = None
            
            # Force garbage collection before recreating
            import gc
            gc.collect()
            
            # Log diagnostic info about memory before creating new instance
            process = psutil.Process()
            logger.info(f"Memory before TPU reset: {process.memory_percent():.1f}%, "
                       f"FDs: {process.num_fds() if hasattr(process, 'num_fds') else 'N/A'}")
            
            # Create new instance
            coral_engine = EdgeTPUInference(model_path, device_type=device_type)
            
            with tpu_stats_lock:
                tpu_stats['reset_count'] += 1
                tpu_stats['last_reset_time'] = time.time()
                tpu_stats['consecutive_errors'] = 0
            
            logger.info("TPU successfully reset")
            update_tpu_success()
            return True
        finally:
            monitored_lock_release(tpu_lock, operation_name="TPU_RESET")
    except Exception as e:
        logger.error(f"Failed to reset TPU: {e}")
        return False

def track_request_start():
    """Track the start of a new request with proper synchronization"""
    global _request_counter, _request_stats_lock, _active_requests, _request_start_times
    global active_requests
    
    with _request_stats_lock:
        index = _request_counter
        _request_counter += 1
        _active_requests.add(index)
        _request_start_times[index] = time.time()
    
    # Update the active_requests counter
    with active_requests_lock:
        active_requests += 1
    
    return index

def track_request_end(index, success=True, duration=None):
    """Track the end of a request with proper counter updates"""
    global _request_stats_lock, _total_requests, _successful_requests, _active_requests
    global _request_start_times, _request_durations
    global active_requests
    
    with _request_stats_lock:
        if index in _active_requests:
            _active_requests.remove(index)
            
        if duration is None and index in _request_start_times:
            duration = time.time() - _request_start_times[index]
            _request_start_times.pop(index, None)
        
        _total_requests += 1  # Increment total requests
        if success:
            _successful_requests += 1  # Increment successful requests
            
        if duration is not None:
            _request_durations.append(duration)
            # Keep only last 1000 durations
            if len(_request_durations) > 1000:
                _request_durations.pop(0)
    
    # Update the active_requests counter
    with active_requests_lock:
        active_requests -= 1

def log_request_stats():
    """Log request statistics with proper synchronization"""
    global _request_stats_lock, _total_requests, _successful_requests, _active_requests
    global _request_durations
    
    with _request_stats_lock:
        total = _total_requests
        successful = _successful_requests
        active = len(_active_requests)
        
        if _request_durations:
            avg_duration = sum(_request_durations) / len(_request_durations)
        else:
            avg_duration = 0
            
        success_rate = (successful / total * 100) if total > 0 else 100
        
        logger.info(
            f"Request Stats: {total} completed, {success_rate:.1f}% success rate, "
            f"avg duration: {avg_duration:.2f}s, currently active: {active}"
        )

@app.route("/health", methods=["GET"])
def health_check():
    try:
        # Update heartbeat on every health check
        update_heartbeat()
        
        # Health checks don't count as API activity for TPU monitoring
        
        process = psutil.Process(os.getpid())
        
        # Get TPU stats (thread-safe copy)
        with tpu_stats_lock:
            tpu_statistics = dict(tpu_stats)
            
        # Calculate TPU success rate
        tpu_success_rate = 0
        if tpu_statistics['inference_count'] > 0:
            tpu_success_rate = (tpu_statistics['successful_inferences'] / 
                               tpu_statistics['inference_count']) * 100
        
        # Calculate request success rate
        request_success_rate = 0
        with request_history_lock:
            completed = [r for r in request_history if r['end_time'] is not None]
            if completed:
                successful = [r for r in completed if r['success']]
                request_success_rate = (len(successful) / len(completed)) * 100
        
        # Get lock stats
        with tpu_lock_state_lock:
            lock_owner = tpu_lock_owner
            lock_held_time = time.time() - tpu_lock_acquired_time if tpu_lock_acquired_time > 0 else 0
        
        stats = {
            "status": "healthy",
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 0,
            "thread_count": process.num_threads(),
            "active_requests": active_requests,
            "tpu_last_success": time.time() - tpu_last_success,
            "tpu_lock": {
                "owner": lock_owner,
                "held_for_seconds": lock_held_time
            },
            "tpu_stats": {
                "inference_count": tpu_statistics['inference_count'],
                "success_rate": tpu_success_rate,
                "reset_count": tpu_statistics['reset_count'],
                "consecutive_errors": tpu_statistics['consecutive_errors'],
                "longest_inference": tpu_statistics['longest_inference_time'],
                "deadlocks_detected": tpu_statistics['deadlocks_detected']
            },
            "request_stats": {
                "active": active_requests,
                "success_rate": request_success_rate
            }
        }
        return flask.jsonify(stats)
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return flask.jsonify({"status": "unhealthy", "error": str(e)}), 500

def watchdog_monitor():
    """Monitor health and TPU status periodically for diagnostics"""
    consecutive_failures = 0
    while True:
        try:
            # Always update the heartbeat to show watchdog is alive
            update_heartbeat()
            
            # Monitor for stuck requests
            monitor_active_requests()
            
            # Run diagnostics instead of resetting
            check_tpu_issues()
            
            # Periodically log request stats
            log_request_stats()
            
            # Check if TPU needs reset only if we have persistent issues
            with tpu_stats_lock:
                if tpu_stats['consecutive_errors'] >= 3:
                    logger.warning(f"Detected {tpu_stats['consecutive_errors']} consecutive TPU errors")
                    reset_tpu_if_needed()
                    
            # Check for potential deadlocks
            with tpu_lock_state_lock:
                if tpu_lock_owner and time.time() - tpu_lock_acquired_time > TPU_OPERATION_TIMEOUT:
                    logger.error(f"Potential deadlock: TPU lock held by {tpu_lock_owner} "
                                f"for {time.time() - tpu_lock_acquired_time:.1f}s")
                    dump_all_threads()
                    # Force reset if it's been too long
                    if time.time() - tpu_lock_acquired_time > TPU_OPERATION_TIMEOUT * 3:
                        logger.critical(f"Deadlock detected! Lock held for {time.time() - tpu_lock_acquired_time:.1f}s")
                        
            # Only reset on health check if we've had multiple failures
            is_healthy = health_monitor.check_health()
            if not is_healthy:
                consecutive_failures += 1
                logger.error(f"Health check failed {consecutive_failures} times")
                
                # Log detailed state
                process = psutil.Process()
                logger.error(f"Process state - "
                           f"Memory: {process.memory_percent():.1f}%, "
                           f"CPU: {process.cpu_percent()}%, "
                           f"FDs: {process.num_fds() if hasattr(process, 'num_fds') else 'N/A'}, "
                           f"Threads: {process.num_threads()}, "
                           f"Active Requests: {active_requests}")
            else:
                if consecutive_failures > 0:  # Log recovery
                    logger.info(f"Health check recovered after {consecutive_failures} failures")
                consecutive_failures = 0
                
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
        
        time.sleep(5)  # Check more frequently (every 5 seconds)

def detect_coral_device():
    """Detect and configure Coral TPU device"""
    try:
        # Try to detect PCI Edge TPU
        pci_path = "/sys/bus/pci/devices/"
        for device in os.listdir(pci_path):
            with open(os.path.join(pci_path, device, 'vendor'), 'r') as f:
                vendor_id = f.read().strip()
            with open(os.path.join(pci_path, device, 'device'), 'r') as f:
                device_id = f.read().strip()
            
            # Global Unichip Corp. vendor ID and Edge TPU device ID
            if vendor_id == '0x1ac1':  # Global Unichip Corp.
                logger.info("Detected PCI Edge TPU (M.2)")
                return "pci"
                
        # Check for USB Edge TPU
        usb_path = "/sys/bus/usb/devices/"
        for device in os.listdir(usb_path):
            vendor_path = os.path.join(usb_path, device, 'idVendor')
            product_path = os.path.join(usb_path, device, 'idProduct')
            
            if os.path.exists(vendor_path) and os.path.exists(product_path):
                with open(vendor_path, 'r') as f:
                    vendor_id = f.read().strip()
                with open(product_path, 'r') as f:
                    product_id = f.read().strip()
                
                # Google's USB vendor ID and Edge TPU product ID
                if vendor_id == '18d1' and product_id == '9302':
                    logger.info("Detected USB Edge TPU")
                    return "usb"
        
        raise RuntimeError("No Edge TPU device found")
        
    except Exception as e:
        logger.error(f"Error detecting Edge TPU: {e}")
        raise

# At the start of the file, after imports
# Check if we're in the correct directory
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'reef_helpers.py')):
    raise RuntimeError("reef_helpers.py not found in the current directory")

# Ensure log directory exists
log_dir = '/var/log/reef'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o755)

# Load model and labels
try:
    labels = load_labels(LABELS)
    device_type = detect_coral_device()
    coral_engine = EdgeTPUInference(MODEL, device_type=device_type)
    # Initialize TPU success timestamp
    update_tpu_success()
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    sys.exit(1)

# Initialize monitors after coral engine
health_monitor = HealthMonitor(coral_engine, logger)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Only start watchdog AFTER health_monitor is initialized
watchdog_thread = threading.Thread(target=watchdog_monitor, daemon=True)
watchdog_thread.start()

# Notify systemd we're ready
notifier = sdnotify.SystemdNotifier()
notifier.notify('READY=1')

# Add a periodic watchdog notification
def watchdog_notify():
    while True:
        try:
            notifier.notify('WATCHDOG=1')
            time.sleep(30)  # Notify every 30 seconds
        except Exception as e:
            logger.error(f"Watchdog notification failed: {e}")

# Start watchdog thread after initialization
import threading
watchdog_thread = threading.Thread(target=watchdog_notify, daemon=True)
watchdog_thread.start()

def _handle_prediction(request_data, request_files, request_id=None):
    perf = PerformanceMonitor()
    monitor = RequestMonitor()
    data = {"success": True}
    data["predictions"] = []
    tpu_inference_time = None

    try:
        resources = monitor.check_resources()
        if resources.get('memory_percent', 0) > 90:
            logger.warning("High memory usage detected!")
        
        perf.start()
        
        image_file = request_files["image"].read()
        image = np.frombuffer(image_file, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        perf.checkpoint("read_file")
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Get image dimensions for logging
        img_height, img_width = image.shape[:2]
        logger.debug(f"Processing image: {img_width}x{img_height} pixels")
        
        # Track TPU operation timing
        tpu_operation_start = time.time()
        
        # Variables to hold results outside the lock
        inference_results = None
        inference_error = None
        tpu_inference_duration = 0
        
        # Single-threaded TPU operation with enhanced deadlock protection
        acquired = monitored_lock_acquire(tpu_lock, timeout=TPU_OPERATION_TIMEOUT, 
                                         operation_name="TPU_INFERENCE")
        if not acquired:
            error_msg = "Failed to acquire TPU lock - possible deadlock"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            # Inside the try...finally to ensure lock release
            try:
                tpu_start = time.time()
                
                # Set a timer to detect hanging TPU operations
                def timeout_handler():
                    logger.critical(f"TPU operation timeout after {TPU_OPERATION_TIMEOUT}s")
                    dump_all_threads()
                
                # Start timer in a separate thread
                timer = threading.Timer(TPU_OPERATION_TIMEOUT, timeout_handler)
                timer.daemon = True
                timer.start()
                
                # Perform actual inference
                boxes, classes, scores = coral_engine.predict(
                    image, 
                    threshold=float(request_data.get('min_confidence', 0.4))
                )
                
                # Cancel timeout handler if we get here
                timer.cancel()
                
                # Calculate inference time
                tpu_inference_duration = time.time() - tpu_start
                
                # Store results to use after releasing the lock
                inference_results = (boxes, classes, scores)
                
                # Just update the timestamp while holding the TPU lock
                # Don't try to acquire other locks while holding this one
                update_tpu_success()
                
            except Exception as e:
                error_time = time.time() - tpu_operation_start
                logger.error(f"TPU prediction error after {error_time:.2f}s: {e}")
                inference_error = e
                tpu_inference_duration = error_time
                raise
                
        finally:
            # Always release the TPU lock before acquiring any other locks
            # This is the key to preventing deadlocks
            monitored_lock_release(tpu_lock, operation_name="TPU_INFERENCE")
        
        # Now that we've released the TPU lock, we can safely update stats
        # which may require acquiring the tpu_stats_lock
        if inference_results:
            # Update success statistics - now safe because TPU lock is released
            update_tpu_stats(success=True, inference_time=tpu_inference_duration)
            
            # Unpack the results
            boxes, classes, scores = inference_results
            
            # Log extended diagnostics for long operations
            if tpu_inference_duration > 5:  # Log details for operations over 5 seconds
                logger.warning(f"Long TPU inference: {tpu_inference_duration:.2f}s, "
                              f"image size: {img_width}x{img_height}, "
                              f"detections: {len(scores)}")
        else:
            # Update failure statistics - now safe because TPU lock is released
            update_tpu_stats(success=False, inference_time=tpu_inference_duration, error=inference_error)
            
            # Schedule TPU reset in background if needed
            if check_tpu_issues():
                threading.Thread(target=reset_tpu_if_needed, daemon=True).start()
            
            if inference_error:
                raise RuntimeError(f"TPU prediction error: {inference_error}")
            else:
                raise RuntimeError("Unknown error during TPU prediction")
        
        perf.checkpoint("inference")
        
        # Process predictions
        for i in range(len(scores)):
            try:
                class_id_int = int(classes[i])
                if class_id_int not in labels:
                    logging.warning(f"Unknown class ID: {class_id_int}, skipping")
                    continue
                    
                prediction = {
                    'confidence': float(scores[i]),
                    'label': labels[class_id_int],
                    'y_min': int(boxes[i][0] * image.shape[0]),
                    'x_min': int(boxes[i][1] * image.shape[1]),
                    'y_max': int(boxes[i][2] * image.shape[0]),
                    'x_max': int(boxes[i][3] * image.shape[1])
                }
                data["predictions"].append(prediction)
                logger.debug(f"Found {labels[class_id_int]} with confidence {scores[i]:.2f}")
            except Exception as e:
                logging.error(f"Error processing prediction {i}: {e}")
                continue
                
        # Modified log message to be more clear
        detection_count = len(data["predictions"])
        if detection_count == 1:
            logger.info(f"Found 1 valid detection with confidence above {float(request_data.get('min_confidence', 0.4)):.2f}")
        else:
            logger.info(f"Found {detection_count} valid detections above confidence threshold")
        
        logger.info(f"Inference completed in {tpu_inference_duration*1000:.2f}ms")
        logger.debug(f"Found {len(data['predictions'])} objects")
        
        # Clean up to avoid memory issues
        del image
        del image_file
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        data["success"] = False
        data["error"] = str(e)

    return data

@app.route("/v1/vision/detection", methods=["POST"])
def predict():
    """Flask route for prediction with better request handling"""
    request_index = track_request_start()
    start_time = time.time()
    success = False
    
    # Update API activity timestamp
    track_api_activity()
    
    try:
        # Try to acquire semaphore with timeout instead of failing immediately
        acquired = request_semaphore.acquire(timeout=5)  # 5 second timeout
        if not acquired:
            logger.warning("Request rejected due to high load")
            return flask.jsonify({
                "success": False,
                "error": "Server is currently at maximum capacity. Please try again shortly."
            }), 503  # Service Unavailable
        
        try:
            # Validate request has the required file
            if 'image' not in flask.request.files:
                return flask.jsonify({
                    "success": False,
                    "error": "No image file provided"
                }), 400
                
            result = _handle_prediction(
                dict(flask.request.form),
                flask.request.files,
                request_id=request_index
            )
            
            success = result.get("success", False)
            return flask.jsonify(result)
            
        finally:
            # Always release the semaphore
            request_semaphore.release()
            
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return flask.jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        # Update request tracking
        duration = time.time() - start_time
        track_request_end(request_index, success=success, duration=duration)

@app.route("/health/detailed", methods=["GET"])
def detailed_health_check():
    """Return detailed health status information with enhanced diagnostics"""
    is_healthy = health_monitor.check_health()
    
    # Get more detailed stats
    process = psutil.Process(os.getpid())
    
    # Get TPU stats (thread-safe copy)
    with tpu_stats_lock:
        tpu_statistics = dict(tpu_stats)
    
    # Calculate TPU success rate
    tpu_success_rate = 0
    if tpu_statistics['inference_count'] > 0:
        tpu_success_rate = (tpu_statistics['successful_inferences'] / 
                           tpu_statistics['inference_count']) * 100
    
    # Calculate average inference time
    avg_inference_time = 0
    if tpu_statistics['inference_times']:
        avg_inference_time = sum(tpu_statistics['inference_times']) / len(tpu_statistics['inference_times'])
    
    # Get API activity information
    with api_request_lock:
        time_since_last_api = time.time() - last_api_request_time
    
    # Get request statistics
    with request_history_lock:
        # Get most recent requests (last 5 minutes)
        five_min_ago = time.time() - 300
        recent_requests = [r for r in request_history if r['start_time'] >= five_min_ago]
        
        completed_requests = [r for r in recent_requests if r['end_time'] is not None]
        successful_requests = [r for r in completed_requests if r['success']]
        
        request_success_rate = 0
        if completed_requests:
            request_success_rate = (len(successful_requests) / len(completed_requests)) * 100
            
        # Calculate average and max duration for completed requests
        avg_duration = 0
        max_duration = 0
        if completed_requests:
            durations = [r['duration'] for r in completed_requests if r['duration'] is not None]
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
    
    stats = {
        "status": "healthy" if is_healthy else "unhealthy",
        "memory_percent": process.memory_percent(),
        "cpu_percent": process.cpu_percent(),
        "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 0,
        "thread_count": process.num_threads(),
        "api_activity": {
            "seconds_since_last_request": time_since_last_api
        },
        "tpu_health": {
            "last_success_seconds_ago": time.time() - tpu_last_success,
            "inference_count": tpu_statistics['inference_count'],
            "success_rate": tpu_success_rate,
            "avg_inference_time": avg_inference_time,
            "max_inference_time": tpu_statistics['longest_inference_time'],
            "reset_count": tpu_statistics['reset_count'],
            "consecutive_errors": tpu_statistics['consecutive_errors'],
            "last_error": tpu_statistics['last_error'],
            "last_error_time_ago": time.time() - tpu_statistics['last_error_time'] if tpu_statistics['last_error_time'] > 0 else None
        },
        "request_stats": {
            "active_requests": active_requests,
            "recent_requests": len(recent_requests),
            "success_rate": request_success_rate,
            "avg_duration": avg_duration,
            "max_duration": max_duration
        }
    }
    
    return flask.jsonify(stats), 200 if is_healthy else 500

@app.route("/health/tpu", methods=["GET"])
def tpu_health_check():
    """Return detailed TPU health information"""
    # Get TPU stats (thread-safe copy)
    with tpu_stats_lock:
        tpu_statistics = dict(tpu_stats)
    
    # Calculate success rate
    success_rate = 0
    if tpu_statistics['inference_count'] > 0:
        success_rate = (tpu_statistics['successful_inferences'] / 
                       tpu_statistics['inference_count']) * 100
    
    # Calculate average inference time
    avg_time = 0
    recent_times = []
    if tpu_statistics['inference_times']:
        avg_time = sum(tpu_statistics['inference_times']) / len(tpu_statistics['inference_times'])
        recent_times = tpu_statistics['inference_times'][-10:]  # Last 10 inference times
    
    stats = {
        "device_type": coral_engine.device_type if coral_engine else "unknown",
        "last_success_seconds_ago": time.time() - tpu_last_success,
        "inference_count": tpu_statistics['inference_count'],
        "success_rate": success_rate,
        "successful_inferences": tpu_statistics['successful_inferences'],
        "failed_inferences": tpu_statistics['failed_inferences'],
        "reset_count": tpu_statistics['reset_count'],
        "avg_inference_time": avg_time,
        "longest_inference_time": tpu_statistics['longest_inference_time'],
        "consecutive_errors": tpu_statistics['consecutive_errors'],
        "recent_inference_times": recent_times,
        "last_error": tpu_statistics['last_error'],
        "last_reset_time_ago": time.time() - tpu_statistics['last_reset_time'] if tpu_statistics['last_reset_time'] > 0 else None
    }
    
    return flask.jsonify(stats)

@app.route("/admin/reset-tpu", methods=["POST"])
def admin_reset_tpu():
    """Admin endpoint to reset the TPU"""
    try:
        logger.info("Manual TPU reset requested")
        if reset_tpu_if_needed():
            return flask.jsonify({"success": True, "message": "TPU reset successfully"}), 200
        else:
            return flask.jsonify({"success": False, "message": "TPU reset failed"}), 500
    except Exception as e:
        logger.error(f"Error resetting TPU: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500

@app.before_request
def log_request_start():
    """Log every request before processing"""
    flask.g.start_time = time.time()
    logger.info(f"Incoming {flask.request.method} request to {flask.request.path} from {flask.request.remote_addr}")

@app.after_request
def log_request_end(response):
    """Log every request after processing, including failures"""
    # Calculate request duration
    if hasattr(flask.g, 'start_time'):
        duration = time.time() - flask.g.start_time
    else:
        duration = 0
        
    # Log the response
    logger.info(
        f"Completed {flask.request.method} {flask.request.path} with status {response.status_code} "
        f"in {duration*1000:.2f}ms (size: {len(response.get_data()) if response.get_data() else 0} bytes)"
    )
    
    # Log detailed error information for non-200 responses
    if response.status_code != 200:
        logger.error(
            f"Request failed: {flask.request.method} {flask.request.path} returned {response.status_code}\n"
            f"Remote IP: {flask.request.remote_addr}\n"
            f"Headers: {dict(flask.request.headers)}\n"
            f"Response: {response.get_data().decode('utf-8', errors='replace')}"
        )

    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Log any unhandled exceptions"""
    logger.error(
        f"Unhandled exception in {flask.request.method} {flask.request.path}",
        exc_info=True
    )
    return {
        "success": False,
        "error": str(e)
    }, 500

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        '/opt/reef_server/models',
        '/var/log/reef'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o755)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

# Set log level based on environment
log_level = logging.DEBUG if os.getenv('ENV') == 'development' else logging.INFO
logger.setLevel(log_level)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Coral TPU Detection Server")
        parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
        parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
        args = parser.parse_args()

        # Start the Flask server
        app.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
