import docker
import subprocess
import time

def ensure_docker_running():
    """Check if Docker Desktop is running and try to start it if not."""
    try:
        client = docker.from_env()
        client.ping()  # Will raise exception if daemon isn't running
        return True
    except docker.errors.DockerException:
        print("Docker Desktop is not running!")
        print("Please start Docker Desktop manually before continuing.")
        print("Once Docker Desktop is running, retry the operation.")
        return False

def start_gentle_container(input_dir):
    """Start the Gentle Docker container with the specified input directory.
    
    Args:
        input_dir (str): Path to the directory containing audio files
        
    Returns:
        bool: True if container started successfully, False otherwise
    """
    docker_command = [
        "docker", "run", 
        "-d",  # Run in detached mode
        "-p", "8765:8765",  # Port mapping
        "-v", f"{input_dir}:/audio",  # Volume mapping with dynamic path
        "lowerquality/gentle"
    ]
    
    try:
        result = subprocess.run(
            docker_command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("Docker container started successfully!")
        print("Container ID:", result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print("Error starting the Docker container.")
        print("Error message:", e.stderr)
        return False
