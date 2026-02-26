"""
Docker Code Executor - Secure Python code execution in isolated containers
"""
import docker
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import traceback


class DockerCodeExecutor:
    """Manages Docker container for secure code execution"""

    def __init__(self, image_name: str = "preprocessing-sandbox:latest"):
        """
        Initialize Docker executor

        Args:
            image_name: Name of Docker image to use
        """
        try:
            self.client = docker.from_env()
            self.image_name = image_name
            self.image_built = False
            print(f"[INFO] Docker client initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Docker client: {str(e)}")
            print(f"[ERROR] Make sure Docker Desktop is running!")
            raise

    def build_image(self, dockerfile_path: str = None) -> bool:
        """
        Build Docker image from Dockerfile

        Args:
            dockerfile_path: Path to directory containing Dockerfile

        Returns:
            True if successful, False otherwise
        """
        try:
            if dockerfile_path is None:
                dockerfile_path = str(Path(__file__).parent)

            print(f"[INFO] Building Docker image '{self.image_name}' from {dockerfile_path}...")

            # Build image
            image, build_logs = self.client.images.build(
                path=dockerfile_path,
                tag=self.image_name,
                rm=True,  # Remove intermediate containers
                forcerm=True
            )

            # Print build logs
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())

            self.image_built = True
            print(f"[SUCCESS] Docker image built successfully!")
            return True

        except docker.errors.BuildError as e:
            print(f"[ERROR] Failed to build Docker image: {str(e)}")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error building image: {str(e)}")
            traceback.print_exc()
            return False

    def check_image_exists(self) -> bool:
        """Check if Docker image exists"""
        try:
            self.client.images.get(self.image_name)
            return True
        except docker.errors.ImageNotFound:
            return False

    def execute_code(self, code: str, dataset_dir: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute Python code in Docker container

        Args:
            code: Python code to execute
            dataset_dir: Directory containing datasets (will be mounted to /workspace/data)
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with execution results:
            {
                'success': bool,
                'stdout': str,
                'stderr': str,
                'error': str (if failed)
            }
        """
        try:
            # Check if image exists, build if not
            if not self.check_image_exists():
                print(f"[INFO] Image not found. Building image...")
                if not self.build_image():
                    return {
                        'success': False,
                        'stdout': '',
                        'stderr': '',
                        'error': 'Failed to build Docker image'
                    }

            # Ensure dataset directory exists
            dataset_path = Path(dataset_dir).absolute()
            if not dataset_path.exists():
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': '',
                    'error': f'Dataset directory does not exist: {dataset_path}'
                }

            print(f"[INFO] Executing code in Docker container...")
            print(f"[INFO] Mounted directory: {dataset_path}")

            # Create a temporary Python file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_script = f.name

            try:
                # Read the script content
                with open(temp_script, 'r') as f:
                    script_content = f.read()

                # Mount volumes
                volumes = {
                    str(dataset_path): {
                        'bind': '/workspace/data',
                        'mode': 'rw'
                    }
                }

                # Run container with resource limits
                container = self.client.containers.run(
                    self.image_name,
                    command=['python3', '-c', script_content],
                    volumes=volumes,
                    remove=True,  # Auto-remove container after execution
                    detach=False,  # Wait for completion
                    stdout=True,
                    stderr=True,
                    mem_limit='1g',  # Limit memory to 1GB
                    cpu_period=100000,
                    cpu_quota=100000,  # Limit to 1 CPU
                    network_disabled=True,  # No internet access for security
                    working_dir='/workspace',
                    user='sandbox',  # Run as non-root user
                    timeout=timeout
                )

                # Container returns combined stdout/stderr as bytes
                output = container.decode('utf-8')

                print(f"[SUCCESS] Code executed successfully!")

                return {
                    'success': True,
                    'stdout': output,
                    'stderr': '',
                    'error': None
                }

            finally:
                # Clean up temporary file
                Path(temp_script).unlink(missing_ok=True)

        except docker.errors.ContainerError as e:
            error_msg = f"Container execution error: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                'stderr': e.stderr.decode('utf-8') if e.stderr else '',
                'error': error_msg
            }

        except docker.errors.APIError as e:
            error_msg = f"Docker API error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'error': error_msg
            }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'error': error_msg
            }

    def cleanup(self):
        """Clean up Docker resources"""
        try:
            # Remove dangling containers
            containers = self.client.containers.list(filters={'status': 'exited'})
            for container in containers:
                if self.image_name in container.image.tags:
                    container.remove()
                    print(f"[INFO] Removed container: {container.id[:12]}")
        except Exception as e:
            print(f"[WARNING] Error during cleanup: {str(e)}")


# Test function
def test_executor():
    """Test the Docker executor"""
    print("="*80)
    print("TESTING DOCKER CODE EXECUTOR")
    print("="*80)

    executor = DockerCodeExecutor()

    # Build image
    if not executor.build_image():
        print("[ERROR] Failed to build image")
        return

    # Test code
    test_code = """
import pandas as pd
import numpy as np

print("Hello from Docker container!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# List files in data directory
import os
print(f"Files in /workspace/data: {os.listdir('/workspace/data')}")
"""

    # Execute
    result = executor.execute_code(
        code=test_code,
        dataset_dir=str(Path(__file__).parent.parent / "datasets")
    )

    print("\nEXECUTION RESULT:")
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['stdout']}")
    if result['error']:
        print(f"Error: {result['error']}")

    # Cleanup
    executor.cleanup()


if __name__ == "__main__":
    test_executor()