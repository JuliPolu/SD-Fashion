import torch

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the current CUDA device ID
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device ID: {current_device}")

        # Get the name of the current CUDA device
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA device name: {device_name}")

        # Get the CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")

        # Get the number of CUDA devices
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    check_cuda()
