import os
import requests
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import argparse
from tqdm import tqdm
import numpy as np
import psutil
from datetime import datetime
import csv
import threading
import random
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format
)
logger = logging.getLogger(__name__)

class BenchmarkTester:
    def __init__(self, url="http://localhost:5000"):
        self.server_url = url
        self.image_dir = "test_images"
        self.results = []
        self.logger = logging.getLogger(__name__)
        self.cpu_usage = []
        self.memory_usage = []
        self.start_time = None
        self.system_metrics = []
        self.test_running = False
        
        # Initialize test images
        self.download_test_images()

    def collect_system_metrics(self):
        """Collect basic system metrics during test"""
        process = psutil.Process()
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(process.memory_percent())

    def download_test_images(self, min_images=20):
        """Download test images if we don't have enough"""
        self.image_dir = "test_images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        # Check existing images first
        existing_images = [f for f in os.listdir(self.image_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(existing_images)} existing images")
        
        if len(existing_images) >= min_images:
            logger.info(f"Using {len(existing_images)} existing test images")
            return

        # Wikimedia Commons and other reliable sources
        image_urls = [
            # Wikimedia Commons images (very stable)
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Pug_600.jpg/800px-Pug_600.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/FC_Bayern_München_Champions_League_Finale_2012.JPG/800px-FC_Bayern_München_Champions_League_Finale_2012.JPG",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/800px-Image_created_with_a_mobile_phone.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/PlazaAltaSantander.jpg/800px-PlazaAltaSantander.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Giant_Panda_2004-03-2.jpg/800px-Giant_Panda_2004-03-2.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Appearance_of_sky_for_weather_forecast%2C_Dhaka%2C_Bangladesh.JPG/800px-Appearance_of_sky_for_weather_forecast%2C_Dhaka%2C_Bangladesh.JPG",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Mostar_Old_Town_Panorama_2007.jpg/800px-Mostar_Old_Town_Panorama_2007.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Garden_bench_001.jpg/800px-Garden_bench_001.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            
            # TensorFlow test images (reliable)
            "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg",
            "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg",
            
            # Additional Wikimedia images
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg/800px-The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Female_police_officers_in_Dhaka%2C_Bangladesh_%2815156006663%29.jpg/800px-Female_police_officers_in_Dhaka%2C_Bangladesh_%2815156006663%29.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Hopetoun_falls.jpg/800px-Hopetoun_falls.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Biandintz_eta_zaldiak_-_modified2.jpg/800px-Biandintz_eta_zaldiak_-_modified2.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/African_Bush_Elephant.jpg/800px-African_Bush_Elephant.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Train_Station_NYC_HDR.jpg/800px-Train_Station_NYC_HDR.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/20170721_Gotland_0099.jpg/800px-20170721_Gotland_0099.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Tennis_Racket_and_Balls.jpg/800px-Tennis_Racket_and_Balls.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Bicycle_parking_lot_at_HAW_Hamburg.jpg/800px-Bicycle_parking_lot_at_HAW_Hamburg.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/N.Tesla.JPG/800px-N.Tesla.JPG",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Subway_car_interior.jpg/800px-Subway_car_interior.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/800px-President_Barack_Obama.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg"
        ]
        
        logger.info(f"Need to download {min_images - len(existing_images)} more images...")
        successful = len(existing_images)  # Start counting from existing images
        
        for url in tqdm(image_urls, desc="Downloading images"):
            try:
                filename = os.path.join(self.image_dir, f"test_image_{successful+1}.jpg")
                if not os.path.exists(filename):
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=10, stream=True)
                    
                    if response.status_code == 200:
                        # Verify it's an image
                        content_type = response.headers.get('content-type', '')
                        if not content_type.startswith('image/'):
                            logger.error(f"Not an image: {url} (Content-Type: {content_type})")
                            continue
                            
                        # Save the image
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        successful += 1
                        logger.info(f"Downloaded image {successful}/{min_images}: {url}")
                    else:
                        logger.error(f"Failed to download {url}: HTTP {response.status_code}")
                else:
                    logger.debug(f"Skipping existing file: {filename}")
                    
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                continue
                
            # Break if we have enough images
            if successful >= min_images:
                break
                
        logger.info(f"Successfully downloaded {successful - len(existing_images)} new images")
        
        # Verify we have enough images
        images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(images)
        
        if total_images < min_images:
            logger.error(f"Failed to download enough images. Only have {total_images}/{min_images}")
            logger.error("Please check your internet connection and try again")
            raise RuntimeError(f"Only found {total_images} images, need at least {min_images}!")
        else:
            logger.info(f"Found {total_images} total images in {self.image_dir}")

    def run_single_test(self, image_path, confidence=0.4):
        """Run a single inference test with detailed metrics"""
        try:
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'min_confidence': confidence}
                response = requests.post(
                    f"{self.server_url}/v1/vision/detection",
                    files=files,
                    data=data
                )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'duration': duration,
                    'predictions': len(result.get('predictions', [])),
                    'prediction_details': result.get('predictions', []),
                    'image': os.path.basename(image_path),
                    'status_code': response.status_code,
                    'response_size': len(response.content),
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'duration': duration,
                    'error': f"HTTP {response.status_code}",
                    'image': os.path.basename(image_path),
                    'status_code': response.status_code,
                    'response_size': len(response.content),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'image': os.path.basename(image_path),
                'timestamp': time.time()
            }

    def run_benchmark(self, num_iterations=100, concurrent=1, delay=0, mode="normal"):
        """Run benchmark using the specified mode and parameters"""
        self.start_time = time.time()
        self.test_running = True
        
        # Configure test parameters based on mode
        if mode == "quick":
            num_iterations = min(num_iterations, 20)
            concurrent = 1
            delay = 0.5
        elif mode == "stress":
            # Stress mode uses higher concurrency
            concurrent = max(concurrent, 5)  # At least 5 concurrent requests
            delay = 0  # No delay between batches
        elif mode == "endurance":
            # Endurance mode runs longer with moderate load
            num_iterations = max(num_iterations, 500)
            delay = 0.2
        elif mode == "burst":
            # Handle burst mode separately
            return self.run_burst_test(num_iterations, concurrent)
        elif mode == "monitor":
            # Monitor mode runs indefinitely
            return self.run_monitor_test()
        
        # Run the actual benchmark
        return self._run_concurrent_benchmark(num_iterations, concurrent, delay)
    
    def _run_concurrent_benchmark(self, num_iterations=100, concurrent=1, delay=0):
        """Run benchmark with proper concurrency using ThreadPoolExecutor"""
        try:
            images = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                raise RuntimeError("No test images found!")
            
            # Reset metrics
            self.cpu_usage = []
            self.memory_usage = []
            self.results = []
            
            # Start background monitoring
            monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            monitor_thread.start()
            
            # Define worker function
            def worker():
                image_path = random.choice(images)
                result = self.run_single_test(image_path)
                return result
            
            # Run benchmark with progress bar
            with tqdm(total=num_iterations, desc="Running benchmark") as pbar:
                for i in range(0, num_iterations, concurrent):
                    batch_size = min(concurrent, num_iterations - i)
                    batch_results = []
                    
                    # Run batch of concurrent requests
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [executor.submit(worker) for _ in range(batch_size)]
                        
                        for future in as_completed(futures):
                            result = future.result()
                            self.results.append(result)
                            batch_results.append(result)
                            pbar.update(1)
                    
                    # Optional delay between batches
                    if delay > 0:
                        time.sleep(delay)
            
            # End monitoring
            self.test_running = False
            
            # Save and print results
            self.save_and_compare_results()
            self.print_results()
            
            return self.results
            
        except Exception as e:
            self.test_running = False
            logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    def run_burst_test(self, num_iterations=100, max_concurrent=10):
        """Run a burst pattern test with alternating high and low traffic"""
        logger.info(f"Running burst test with {num_iterations} total requests")
        
        try:
            # Reset metrics
            self.cpu_usage = []
            self.memory_usage = []
            self.results = []
            self.start_time = time.time()
            self.test_running = True
            
            # Start background monitoring
            monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            monitor_thread.start()
            
            # Define burst patterns: (concurrent_requests, requests_in_burst, delay_after_burst)
            patterns = [
                (max_concurrent, int(num_iterations * 0.3), 5),  # High traffic burst
                (1, int(num_iterations * 0.1), 2),              # Low traffic period
                (max_concurrent, int(num_iterations * 0.3), 5),  # Another high burst
                (2, int(num_iterations * 0.3), 0)               # Medium traffic to finish
            ]
            
            # Make sure we have all iterations accounted for
            total_pattern_requests = sum(p[1] for p in patterns)
            if total_pattern_requests < num_iterations:
                # Add any remaining to the last pattern
                patterns[-1] = (patterns[-1][0], patterns[-1][1] + (num_iterations - total_pattern_requests), patterns[-1][2])
            
            with tqdm(total=num_iterations, desc="Running burst test") as pbar:
                requests_completed = 0
                
                for pattern_index, (concurrent, requests_in_burst, delay_after) in enumerate(patterns):
                    logger.info(f"Pattern {pattern_index+1}: {concurrent} concurrent requests, {requests_in_burst} total")
                    
                    # Run this burst pattern
                    results = self._run_concurrent_benchmark(
                        num_iterations=requests_in_burst,
                        concurrent=concurrent,
                        delay=0
                    )
                    
                    requests_completed += len(results)
                    pbar.update(len(results))
                    
                    # Apply delay after burst (if not the last pattern)
                    if delay_after > 0 and pattern_index < len(patterns) - 1:
                        logger.info(f"Pausing for {delay_after} seconds...")
                        time.sleep(delay_after)
            
            # End monitoring
            self.test_running = False
            
            # Save and print results
            self.save_and_compare_results()
            self.print_results()
            
            return self.results
            
        except Exception as e:
            self.test_running = False
            logger.error(f"Burst test failed: {str(e)}")
            raise

    def run_monitor_test(self):
        """Run a specialized test to continuously monitor API health"""
        logger.info("Starting continuous API monitoring")
        
        while True:
            try:
                # Make a simple health check request
                health_url = f"{self.server_url}/health"
                health_start = time.time()
                response = requests.get(health_url, timeout=5)
                health_duration = time.time() - health_start
                
                if response.status_code != 200:
                    logger.error(f"API health check failed: HTTP {response.status_code}")
                
                # Collect system metrics
                self.collect_system_metrics()
                
                # Add timestamped metric
                self.system_metrics.append({
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
                    'memory_percent': self.memory_usage[-1] if self.memory_usage else 0,
                    'active_requests': len([r for r in self.results if r.get('timestamp', 0) > time.time() - 5])
                })
                
                # Sleep between checks
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
        
        logger.debug("System monitoring stopped")

    def _monitor_system(self):
        """Background system monitoring"""
        while self.test_running:
            try:
                # Get CPU and memory metrics
                cpu = psutil.cpu_percent()
                process = psutil.Process()
                memory = process.memory_percent()
                
                # Store metrics
                self.cpu_usage.append(cpu)
                self.memory_usage.append(memory)
                
                # Add timestamped metric
                self.system_metrics.append({
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu,
                    'memory_percent': memory,
                    'active_requests': len([r for r in self.results if r.get('timestamp', 0) > time.time() - 5])
                })
                
                # Don't sample too frequently
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
        
        logger.debug("System monitoring stopped")

    def save_results(self):
        """Save detailed results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save test results
        if self.results:
            results_file = f"benchmark_results_{timestamp}.csv"
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
                logger.info(f"Results saved to {results_file}")
            
        # Save system metrics
        if self.system_metrics:
            metrics_file = f"system_metrics_{timestamp}.csv"
            with open(metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.system_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.system_metrics)
                logger.info(f"System metrics saved to {metrics_file}")

    def save_and_compare_results(self):
        """Save results and compare with previous runs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Calculate current run metrics
        successful = [r for r in self.results if r['success']]
        if successful:
            current_metrics = {
                'timestamp': timestamp,
                'average_duration': statistics.mean(r['duration'] for r in successful) * 1000,  # ms
                'successful_requests': len(successful),
                'total_requests': len(self.results),
                'avg_cpu': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'avg_memory': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'error_rate': (len(self.results) - len(successful)) / len(self.results) * 100 if self.results else 0,
                'p95_duration': np.percentile([r['duration'] for r in successful], 95) * 1000 if successful else 0,
                'p99_duration': np.percentile([r['duration'] for r in successful], 99) * 1000 if successful else 0
            }
            
            # Save current results
            results_file = os.path.join(results_dir, f"benchmark_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(current_metrics, f)
                
            # Export detailed results
            detailed_file = os.path.join(results_dir, f"detailed_{timestamp}.csv")
            if self.results:
                with open(detailed_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['timestamp', 'duration', 'success', 'predictions', 'image', 'status_code'])
                    writer.writeheader()
                    for r in self.results:
                        writer.writerow({
                            'timestamp': r.get('timestamp', 0),
                            'duration': r.get('duration', 0),
                            'success': r.get('success', False),
                            'predictions': r.get('predictions', 0),
                            'image': r.get('image', ''),
                            'status_code': r.get('status_code', 0)
                        })
                logger.info(f"Detailed results saved to {detailed_file}")
            
            # Compare with previous runs
            previous_files = sorted(glob.glob(os.path.join(results_dir, "benchmark_*.json")))
            if len(previous_files) > 1:  # More than just our current run
                with open(previous_files[-2]) as f:  # Get the last run before current
                    last_run = json.load(f)
                
                print("\nComparison with previous run:")
                print(f"  Previous avg response: {last_run['average_duration']:.2f}ms")
                print(f"  Current avg response:  {current_metrics['average_duration']:.2f}ms")
                
                diff_pct = ((current_metrics['average_duration'] - last_run['average_duration']) 
                           / last_run['average_duration'] * 100)
                print(f"  Performance change:    {diff_pct:+.1f}% ({'slower' if diff_pct > 0 else 'faster'})")
                
                print(f"  Previous error rate:   {last_run.get('error_rate', 0):.1f}%")
                print(f"  Current error rate:    {current_metrics['error_rate']:.1f}%")
                print(f"  Previous CPU usage:    {last_run['avg_cpu']:.1f}%")
                print(f"  Current CPU usage:     {current_metrics['avg_cpu']:.1f}%")

    def print_results(self):
        """Print comprehensive benchmark results"""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        
        print("\nRequest Summary:")
        print(f"  Total Requests: {len(self.results)}")
        print(f"  Successful:    {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"  Failed:        {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        
        if successful:
            durations = [r['duration'] for r in successful]
            print("\nTiming Statistics:")
            print(f"  Average:  {statistics.mean(durations)*1000:.2f}ms")
            print(f"  Median:   {statistics.median(durations)*1000:.2f}ms")
            print(f"  95th pct: {np.percentile(durations, 95)*1000:.2f}ms")
            print(f"  99th pct: {np.percentile(durations, 99)*1000:.2f}ms")
            print(f"  Min:      {min(durations)*1000:.2f}ms")
            print(f"  Max:      {max(durations)*1000:.2f}ms")
        
        if self.cpu_usage and self.memory_usage:
            print("\nSystem Metrics:")
            print(f"  Avg CPU Usage:    {statistics.mean(self.cpu_usage):.1f}%")
            print(f"  Max CPU Usage:    {max(self.cpu_usage):.1f}%")
            print(f"  Avg Memory Usage: {statistics.mean(self.memory_usage):.1f}%")
            print(f"  Max Memory Usage: {max(self.memory_usage):.1f}%")
        
        # Check for slow periods in the test
        if successful:
            slow_threshold = np.percentile(durations, 95) * 2  # 2x the 95th percentile
            slow_requests = [r for r in successful if r['duration'] > slow_threshold]
            if slow_requests:
                print("\nSlow Periods Analysis:")
                slow_times = [r['timestamp'] for r in slow_requests]
                # Group slow requests that happened close together
                if len(slow_times) > 1:
                    slow_times.sort()
                    clusters = []
                    current_cluster = [slow_times[0]]
                    
                    for t in slow_times[1:]:
                        if t - current_cluster[-1] < 5:  # Within 5 seconds
                            current_cluster.append(t)
                        else:
                            clusters.append(current_cluster)
                            current_cluster = [t]
                    
                    if current_cluster:
                        clusters.append(current_cluster)
                    
                    print(f"  Found {len(slow_requests)} abnormally slow requests (>{slow_threshold*1000:.0f}ms)")
                    print(f"  Grouped into {len(clusters)} slow periods")
                    for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
                        start_time = datetime.fromtimestamp(cluster[0]).strftime("%H:%M:%S")
                        print(f"  - Slow period {i+1}: {start_time}, {len(cluster)} requests, avg: "
                             f"{statistics.mean([r['duration'] for r in slow_requests if r['timestamp'] in cluster])*1000:.1f}ms")
        
        if failed:
            print("\nErrors encountered:")
            error_counts = {}
            for r in failed:
                error = str(r.get('error', 'Unknown error'))
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                print(f"  - {error}: {count} times")
            
            # Check for patterns in errors
            if len(failed) > 5:
                consecutive_errors = 0
                max_consecutive = 0
                for i in range(1, len(self.results)):
                    if not self.results[i]['success'] and not self.results[i-1]['success']:
                        consecutive_errors += 1
                        max_consecutive = max(max_consecutive, consecutive_errors)
                    elif not self.results[i]['success']:
                        consecutive_errors = 1
                    else:
                        consecutive_errors = 0
                
                if max_consecutive > 2:
                    print(f"\n  Alert: Found {max_consecutive} consecutive failures - possible API outage")

    def test_api_responsiveness(self, duration=60):
        """Run a specialized test to check API responsiveness over time"""
        print(f"Testing API responsiveness for {duration} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration
        interval = 1  # 1 second between requests
        results = []
        
        # Setup progress bar
        with tqdm(total=int(duration/interval), desc="Monitoring API") as pbar:
            while time.time() < end_time:
                try:
                    # Make a simple health check request
                    health_url = f"{self.server_url}/health"
                    health_start = time.time()
                    response = requests.get(health_url, timeout=5)
                    health_duration = time.time() - health_start
                    
                    results.append({
                        'timestamp': time.time(),
                        'success': response.status_code == 200,
                        'duration': health_duration,
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    results.append({
                        'timestamp': time.time(),
                        'success': False,
                        'error': str(e),
                        'duration': time.time() - health_start
                    })
                    
                pbar.update(1)
                
                # Sleep until next interval
                next_time = start_time + len(results) * interval
                sleep_time = max(0, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Analyze and print results
        print("\nAPI Responsiveness Results:")
        successful = [r for r in results if r.get('success', False)]
        print(f"  Total checks: {len(results)}")
        print(f"  Successful:   {len(successful)} ({len(successful)/len(results)*100 if results else 0:.1f}%)")
        
        if successful:
            durations = [r['duration'] for r in successful]
            print(f"  Average response time: {statistics.mean(durations)*1000:.2f}ms")
            print(f"  Max response time:     {max(durations)*1000:.2f}ms")
        
        if len(results) != len(successful):
            print("\nError periods:")
            for i, r in enumerate(results):
                if not r.get('success', False):
                    time_str = datetime.fromtimestamp(r['timestamp']).strftime("%H:%M:%S")
                    print(f"  {time_str}: {r.get('error', 'Failed request')}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark the REEF server')
    parser.add_argument('--url', default='http://localhost:5000', help='Server URL')
    parser.add_argument('--iterations', type=int, default=100, help='Number of test iterations')
    parser.add_argument('--concurrent', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--delay', type=float, default=0, help='Delay between request batches (seconds)')
    parser.add_argument('--mode', choices=['quick', 'normal', 'stress', 'endurance', 'burst', 'monitor'], 
                      default='normal', help='Benchmark mode')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Duration in seconds for monitoring mode')
    
    args = parser.parse_args()
    
    tester = BenchmarkTester(url=args.url)
    
    if args.mode == 'monitor':
        print(f"Starting API monitoring for {args.duration} seconds")
        tester.test_api_responsiveness(duration=args.duration)
    else:
        print(f"Starting benchmark in {args.mode} mode:")
        print(f"  Iterations: {args.iterations}")
        print(f"  Max concurrent requests: {args.concurrent}")
        print(f"  Request delay: {args.delay}s")
        
        tester.run_benchmark(
            num_iterations=args.iterations,
            concurrent=args.concurrent,
            delay=args.delay,
            mode=args.mode
        )

if __name__ == "__main__":
    main() 
