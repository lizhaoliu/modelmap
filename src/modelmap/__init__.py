"""modelmap — interactive, animated architecture maps for Hugging Face models."""

__version__ = "0.1.0"

# NOTE: no eager submodule imports here. The server parent process must stay
# torch-free (~250 MB); extraction imports torch only inside worker processes.
