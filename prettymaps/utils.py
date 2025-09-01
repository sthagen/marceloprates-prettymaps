import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if "logging" in kwargs and kwargs["logging"]:
            logging.info(
                f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds"
            )
        return result

    return wrapper
