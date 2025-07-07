# Save as debug_mp.py
import _multiprocessing
import os

print(f"Process ID: {os.getpid()}")

try:
    # Try to create a semaphore directly
    sem = _multiprocessing.SemLock(0, 1, 1, "/test", True)
    print("Direct semaphore creation worked!")
except Exception as e:
    print(f"Direct semaphore failed: {e}")
    print(f"Error type: {type(e)}")
    
# Check errno meaning
import errno
print(f"\nErrno 28 is: {errno.errorcode.get(28, 'unknown')}")