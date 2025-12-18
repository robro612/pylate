#!/usr/bin/env python3
"""
Asynchronous training monitor that evaluates checkpoints as they're generated.

This monitor runs as a lightweight CPU job and launches GPU evaluation jobs
for new checkpoints discovered in a training output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointStatus:
    """Status information for a checkpoint."""
    checkpoint_path: str
    checkpoint_number: int
    status: str  # pending, queued, running, completed, failed
    slurm_job_id: Optional[str] = None
    submission_time: Optional[str] = None
    completion_time: Optional[str] = None
    exit_code: Optional[int] = None
    retry_count: int = 0
    eval_script_path: Optional[str] = None
    log_file: Optional[str] = None


class TrainingMonitor:
    """Monitor training directory and launch evaluation jobs for new checkpoints."""
    
    def __init__(self, config_path: str, model_dir: Optional[str] = None):
        """Initialize the monitor with configuration."""
        self.config = self._load_config(config_path)
        
        # Override training_output_dir if model_dir is provided
        if model_dir:
            self.config['training_output_dir'] = model_dir
            # Recalculate monitor hash with new directory
            training_dir = Path(model_dir).resolve()
            eval_config_str = json.dumps(self.config.get('evaluation', {}), sort_keys=True)
            hash_input = f"{training_dir}_{eval_config_str}"
            monitor_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            self.config['monitor_hash'] = monitor_hash
            # Update state and lock file paths with new hash
            self.config['state_file'] = f'monitor/state/state_{monitor_hash}.json'
            self.config['lock_file'] = f'monitor/state/monitor_{monitor_hash}.lock'
        
        self.state_file = Path(self.config['state_file'])
        self.lock_file = Path(self.config['lock_file'])
        self.monitor_hash = self.config['monitor_hash']
        self.checkpoints: Dict[str, CheckpointStatus] = {}
        self.running = True
        self.eval_script_path = None
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Create necessary directories
        self._setup_directories()
        
        # Acquire lock
        self._acquire_lock()
        
        # Load or initialize state
        self._load_state()
        
        # Generate the evaluation script once at startup
        self.eval_script_path = self._generate_eval_script()
        
        logger.info("Training monitor initialized")
        logger.info(f"Monitoring directory: {self.config['training_output_dir']}")
        logger.info(f"Monitor hash: {self.monitor_hash}")
        logger.info(f"Evaluation script: {self.eval_script_path}")
        logger.info(f"Max concurrent jobs: {self.config['max_concurrent_jobs']}")
        logger.info(f"Polling interval: {self.config['polling_interval']}s")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        config.setdefault('polling_interval', 60)
        config.setdefault('max_concurrent_jobs', 1)
        config.setdefault('max_retries', 2)
        config.setdefault('prioritize_latest', True)
        
        # Generate unique hash based on training directory and config
        # This allows multiple monitors to run for different training directories
        # or different eval configs on the same directory
        training_dir = Path(config['training_output_dir']).resolve()
        
        # Include evaluation config in hash to allow different eval configs on same dir
        eval_config_str = json.dumps(config.get('evaluation', {}), sort_keys=True)
        hash_input = f"{training_dir}_{eval_config_str}"
        monitor_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        config['monitor_hash'] = monitor_hash
        config.setdefault('state_file', f'monitor/state/state_{monitor_hash}.json')
        config.setdefault('lock_file', f'monitor/state/monitor_{monitor_hash}.lock')
        config.setdefault('log_dir', 'monitor/logs')
        config.setdefault('eval_scripts_dir', 'monitor/eval_scripts')
        
        return config
    
    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['eval_scripts_dir']).mkdir(parents=True, exist_ok=True)
        # Ensure state directory exists (for lock and state files)
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _acquire_lock(self):
        """Acquire lock file to prevent multiple monitor instances."""
        if self.lock_file.exists():
            # Check if process is still running
            try:
                with open(self.lock_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if process exists
                try:
                    os.kill(old_pid, 0)
                    logger.error(f"Another monitor instance is running (PID: {old_pid})")
                    sys.exit(1)
                except OSError:
                    # Process doesn't exist, remove stale lock
                    logger.warning(f"Removing stale lock file (PID: {old_pid})")
                    self.lock_file.unlink()
            except (ValueError, FileNotFoundError):
                self.lock_file.unlink()
        
        # Create lock file with current PID
        with open(self.lock_file, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"Acquired lock (PID: {os.getpid()})")
    
    def _release_lock(self):
        """Release lock file."""
        if self.lock_file.exists():
            self.lock_file.unlink()
            logger.info("Released lock")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _load_state(self):
        """Load state from file or initialize new state."""
        if self.state_file.exists():
            logger.info(f"Loading state from {self.state_file}")
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                for checkpoint_path, checkpoint_data in state_data.items():
                    self.checkpoints[checkpoint_path] = CheckpointStatus(**checkpoint_data)
                
                logger.info(f"Loaded state for {len(self.checkpoints)} checkpoints")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                logger.info("Starting with fresh state")
        else:
            logger.info("No existing state found, starting fresh")
    
    def _save_state(self):
        """Save current state to file."""
        try:
            state_data = {
                path: asdict(status) 
                for path, status in self.checkpoints.items()
            }
            
            # Write atomically using temporary file
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            temp_file.replace(self.state_file)
            
            logger.debug("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _discover_checkpoints(self) -> List[str]:
        """Discover checkpoint directories in training output."""
        training_dir = Path(self.config['training_output_dir'])
        
        if not training_dir.exists():
            logger.warning(f"Training directory does not exist: {training_dir}")
            return []
        
        # Find checkpoint directories matching pattern
        checkpoint_pattern = self.config.get('checkpoint_pattern', 'checkpoint-*')
        checkpoints = []
        
        for checkpoint_dir in training_dir.glob(checkpoint_pattern):
            if checkpoint_dir.is_dir():
                # Extract checkpoint number
                match = re.search(r'checkpoint-(\d+)', checkpoint_dir.name)
                if match:
                    checkpoints.append(str(checkpoint_dir.resolve()))
        
        return checkpoints
    
    def _get_checkpoint_number(self, checkpoint_path: str) -> int:
        """Extract checkpoint number from path."""
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        if match:
            return int(match.group(1))
        return 0
    
    def _update_checkpoint_discovery(self):
        """Discover new checkpoints and update state."""
        discovered = self._discover_checkpoints()
        new_count = 0
        
        for checkpoint_path in discovered:
            if checkpoint_path not in self.checkpoints:
                checkpoint_number = self._get_checkpoint_number(checkpoint_path)
                self.checkpoints[checkpoint_path] = CheckpointStatus(
                    checkpoint_path=checkpoint_path,
                    checkpoint_number=checkpoint_number,
                    status='pending'
                )
                new_count += 1
                logger.info(f"Discovered new checkpoint: {checkpoint_path} (step {checkpoint_number})")
        
        if new_count > 0:
            logger.info(f"Discovered {new_count} new checkpoint(s)")
            self._save_state()
    
    def _check_job_status(self, job_id: str) -> tuple[str, Optional[int]]:
        """
        Check SLURM job status.
        
        Returns:
            Tuple of (status, exit_code) where status is 'running', 'completed', or 'failed'
        """
        try:
            # Use sacct to check job status
            result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State,ExitCode', '--noheader', '--parsable2'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to query job {job_id}: {result.stderr}")
                return 'running', None
            
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                # Job not found in sacct yet, check squeue
                result = subprocess.run(
                    ['squeue', '-j', job_id, '--noheader'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return 'running', None
                return 'running', None
            
            # Parse first line (main job status)
            parts = lines[0].split('|')
            if len(parts) >= 2:
                state = parts[0].strip()
                exit_code_str = parts[1].strip()
                
                # Parse exit code (format: exitcode:signal)
                exit_code = None
                if ':' in exit_code_str:
                    try:
                        exit_code = int(exit_code_str.split(':')[0])
                    except ValueError:
                        pass
                
                # Map SLURM states to our states
                if state in ['RUNNING', 'PENDING', 'CONFIGURING']:
                    return 'running', None
                elif state in ['COMPLETED']:
                    return 'completed', exit_code or 0
                elif state in ['FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY']:
                    return 'failed', exit_code or 1
                else:
                    logger.warning(f"Unknown SLURM state for job {job_id}: {state}")
                    return 'running', None
            
            return 'running', None
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout checking status for job {job_id}")
            return 'running', None
        except Exception as e:
            logger.error(f"Error checking job status for {job_id}: {e}")
            return 'running', None
    
    def _update_running_jobs(self):
        """Check status of running jobs and update state."""
        running_checkpoints = [
            (path, status) for path, status in self.checkpoints.items()
            if status.status == 'running' and status.slurm_job_id
        ]
        
        if not running_checkpoints:
            return
        
        logger.debug(f"Checking status of {len(running_checkpoints)} running job(s)")
        
        for checkpoint_path, status in running_checkpoints:
            job_status, exit_code = self._check_job_status(status.slurm_job_id)
            
            if job_status != 'running':
                status.status = job_status
                status.exit_code = exit_code
                status.completion_time = datetime.now().isoformat()
                
                if job_status == 'completed':
                    logger.info(f"Job {status.slurm_job_id} completed for checkpoint {status.checkpoint_number}")
                elif job_status == 'failed':
                    logger.warning(f"Job {status.slurm_job_id} failed for checkpoint {status.checkpoint_number} (exit code: {exit_code})")
                    
                    # Handle retries
                    if status.retry_count < self.config['max_retries']:
                        status.retry_count += 1
                        status.status = 'pending'
                        status.slurm_job_id = None
                        logger.info(f"Retrying checkpoint {status.checkpoint_number} (attempt {status.retry_count + 1}/{self.config['max_retries'] + 1})")
                
                self._save_state()
    
    def _generate_eval_script(self) -> str:
        """
        Generate evaluation wrapper script once at startup.
        The script takes checkpoint path as a single argument.
        """
        script_name = f"eval_monitor_{self.monitor_hash}.sh"
        script_path = Path(self.config['eval_scripts_dir']) / script_name
        
        # Get evaluation configuration
        eval_config = self.config['evaluation']
        datasets = eval_config.get('datasets', ['beir/nfcorpus/test'])
        index_types = eval_config.get('index_types', ['Flat'])
        batch_size = eval_config.get('batch_size', 2048)
        k = eval_config.get('k', 100)
        k_token = eval_config.get('k_token', 4000)
        cache_embeddings = eval_config.get('cache_embeddings', True)
        cache_dir = eval_config.get('cache_dir', 'embedding_cache')
        shard_size = eval_config.get('shard_size', 10000)
        model_dtype = eval_config.get('model_dtype', 'fp32')
        embedding_dtype = eval_config.get('embedding_dtype', 'fp16')
        lowercase = eval_config.get('lowercase', False)
        
        # Build command with $CHECKPOINT_PATH variable
        datasets_str = ' '.join([f'"{d}"' for d in datasets])
        index_types_str = ' '.join(index_types)
        
        cmd_parts = [
            'python eval_model_irds.py',
            '--model_name_or_path "$CHECKPOINT_PATH"',
            f'--dataset_name {datasets_str}',
            f'--index_types {index_types_str}',
            f'--batch_size {batch_size}',
            f'--k {k}',
            f'--k_token {k_token}',
            f'--shard_size {shard_size}',
            f'--model_dtype {model_dtype}',
            f'--embedding_dtype {embedding_dtype}',
        ]
        
        if cache_embeddings:
            cmd_parts.append('--cache_embeddings')
            cmd_parts.append(f'--cache_dir {cache_dir}')
        
        if lowercase:
            cmd_parts.append('--lowercase')
        
        # Add optional parameters
        if 'query_len' in eval_config:
            cmd_parts.append(f'--query_len {eval_config["query_len"]}')
        if 'doc_len' in eval_config:
            cmd_parts.append(f'--doc_len {eval_config["doc_len"]}')
        if 'num_leaves' in eval_config:
            cmd_parts.append(f'--num_leaves {eval_config["num_leaves"]}')
        if 'num_leaves_to_search' in eval_config:
            cmd_parts.append(f'--num_leaves_to_search {eval_config["num_leaves_to_search"]}')
        if 'num_neighbors' in eval_config:
            cmd_parts.append(f'--num_neighbors {eval_config["num_neighbors"]}')
        
        command = ' \\\n    '.join(cmd_parts)
        
        # Get SLURM configuration
        slurm_config = self.config['slurm']
        time_limit = slurm_config.get('time', '4:00:00')
        partition = slurm_config.get('partition', '')
        gres = slurm_config.get('gres', 'gpu:v100:1')
        mem = slurm_config.get('mem', '')
        
        # Generate script content
        script_content = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --gres={gres}
"""
        
        if partition:
            script_content += f"#SBATCH --partition={partition}\n"
        if mem:
            script_content += f"#SBATCH --mem={mem}\n"
        
        script_content += f"""
# Evaluation script for training monitor {self.monitor_hash}
# Generated at {datetime.now().isoformat()}
# Takes checkpoint path as single argument

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "ERROR: Checkpoint path not provided"
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

CHECKPOINT_PATH="$1"
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")

echo "Starting evaluation"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Monitor hash: {self.monitor_hash}"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Run evaluation
{command}

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
"""
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Generated evaluation script: {script_path}")
        return str(script_path)
    
    def _submit_job(self, checkpoint_path: str) -> Optional[str]:
        """Submit evaluation job for a checkpoint."""
        status = self.checkpoints[checkpoint_path]
        
        # Use the shared evaluation script (generated once at startup)
        status.eval_script_path = self.eval_script_path
        
        # Setup log file with monitor hash for uniqueness
        log_file = Path(self.config['log_dir']) / f"eval_{self.monitor_hash}_checkpoint_{status.checkpoint_number}.log"
        status.log_file = str(log_file)
        
        # Submit job with checkpoint path as argument
        job_name = f"eval_{self.monitor_hash[:4]}_ckpt_{status.checkpoint_number}"
        
        try:
            result = subprocess.run(
                [
                    'sbatch',
                    f'--job-name={job_name}',
                    f'--output={log_file}',
                    f'--error={log_file}',
                    self.eval_script_path,
                    checkpoint_path  # Pass checkpoint path as argument
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to submit job: {result.stderr}")
                return None
            
            # Parse job ID from output
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if match:
                job_id = match.group(1)
                logger.info(f"Submitted job {job_id} for checkpoint {status.checkpoint_number}")
                return job_id
            else:
                logger.error(f"Could not parse job ID from: {result.stdout}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout submitting job")
            return None
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def _get_pending_queue(self) -> List[str]:
        """Get list of pending checkpoints, sorted by priority."""
        pending = [
            (path, status) for path, status in self.checkpoints.items()
            if status.status == 'pending'
        ]
        
        if self.config['prioritize_latest']:
            # Sort by checkpoint number descending (latest first)
            pending.sort(key=lambda x: x[1].checkpoint_number, reverse=True)
        else:
            # Sort by checkpoint number ascending (earliest first)
            pending.sort(key=lambda x: x[1].checkpoint_number)
        
        return [path for path, _ in pending]
    
    def _get_running_count(self) -> int:
        """Count number of currently running jobs."""
        return sum(1 for status in self.checkpoints.values() if status.status == 'running')
    
    def _submit_pending_jobs(self):
        """Submit pending jobs if slots are available."""
        running_count = self._get_running_count()
        max_jobs = self.config['max_concurrent_jobs']
        
        if running_count >= max_jobs:
            logger.debug(f"Max concurrent jobs reached ({running_count}/{max_jobs})")
            return
        
        pending_queue = self._get_pending_queue()
        
        if not pending_queue:
            return
        
        # Submit jobs to fill available slots
        slots_available = max_jobs - running_count
        to_submit = pending_queue[:slots_available]
        
        for checkpoint_path in to_submit:
            status = self.checkpoints[checkpoint_path]
            
            logger.info(f"Submitting evaluation for checkpoint {status.checkpoint_number}")
            job_id = self._submit_job(checkpoint_path)
            
            if job_id:
                status.status = 'running'
                status.slurm_job_id = job_id
                status.submission_time = datetime.now().isoformat()
                self._save_state()
            else:
                logger.error(f"Failed to submit job for checkpoint {status.checkpoint_number}")
    
    def _print_status_summary(self):
        """Print summary of current status."""
        status_counts = {}
        for status in self.checkpoints.values():
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        total = len(self.checkpoints)
        pending = status_counts.get('pending', 0)
        running = status_counts.get('running', 0)
        completed = status_counts.get('completed', 0)
        failed = status_counts.get('failed', 0)
        
        logger.info("="*60)
        logger.info("STATUS SUMMARY")
        logger.info(f"Total checkpoints: {total}")
        logger.info(f"  Pending: {pending}")
        logger.info(f"  Running: {running}")
        logger.info(f"  Completed: {completed}")
        logger.info(f"  Failed: {failed}")
        
        if running > 0:
            logger.info("\nRunning jobs:")
            for path, status in self.checkpoints.items():
                if status.status == 'running':
                    logger.info(f"  Checkpoint {status.checkpoint_number}: Job {status.slurm_job_id}")
        
        logger.info("="*60)
    
    def run(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        iteration = 0
        while self.running:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            
            try:
                # Discover new checkpoints
                self._update_checkpoint_discovery()
                
                # Update status of running jobs
                self._update_running_jobs()
                
                # Submit pending jobs if slots available
                self._submit_pending_jobs()
                
                # Print status summary periodically
                if iteration % 5 == 0 or iteration == 1:
                    self._print_status_summary()
                
                # Sleep until next iteration
                if self.running:
                    logger.info(f"Sleeping for {self.config['polling_interval']}s...")
                    time.sleep(self.config['polling_interval'])
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.config['polling_interval'])
        
        logger.info("Monitoring loop stopped")
    
    def shutdown(self):
        """Perform cleanup on shutdown."""
        logger.info("Shutting down monitor...")
        
        # Save final state
        self._save_state()
        
        # Print final summary
        self._print_status_summary()
        
        # Release lock
        self._release_lock()
        
        logger.info("Monitor shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor training directory and evaluate checkpoints asynchronously"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Override training_output_dir from config file'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TrainingMonitor(args.config, model_dir=args.model_dir)
    
    try:
        # Run monitoring loop
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        monitor.shutdown()


if __name__ == '__main__':
    main()

