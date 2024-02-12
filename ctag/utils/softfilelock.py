# Copyright (c) 2024
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import fcntl
import filelock

class MySoftFileLock(filelock.SoftFileLock):
    def __init__(self, lock_file, *args):
        if isinstance(lock_file, int):
            lock_file = str(lock_file)
        elif isinstance(lock_file, os.PathLike):
            lock_file = os.fspath(lock_file)
        super().__init__(lock_file, *args)
        self.fd = None
    def __call__(self, fd, operation):
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            self.acquire(timeout=0)
        elif operation == fcntl.LOCK_EX:
            self.acquire()
        elif operation == fcntl.LOCK_UN:
            self.release()
        else:
            raise ValueError(f'Unsupported flock operation: {operation}')
        self.fd = fd
    def release(self, force=False):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd = None
        super().release()
