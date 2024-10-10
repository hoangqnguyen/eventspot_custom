import time

class Timer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self  # If you need to return any object, it would be here

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"{self.message}: {elapsed_time:.4f} seconds")