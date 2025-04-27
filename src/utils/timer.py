# >>> timer.py
# Original author: Andrea Vincenzo Ricciardi

import time

class Timer:
    """Context manager to measure execution time of a code block.

    Methods
    -------
    __enter__()
        Starts the timer and returns the `Timer` instance.
    __exit__(exc_type, exc_value, traceback)
        Stops the timer and stores the duration as `self.duration`.
    """

    def __enter__(self) -> "Timer":
        """Start the timer at the beginning of the `with` block.

        Returns
        -------
        self : Timer
            The `Timer` instance for context usage.
        """
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        """Stop the timer and calculate the duration when exiting the `with` block.

        Parameters
        ----------
        exc_type : type or None
            Exception type, if raised.
        exc_value : BaseException or None
            Exception value, if raised.
        traceback : traceback or None
            Traceback object, if an exception occurred.
        """
        self.end = time.time()
        self.duration = round(self.end - self.start, 4)