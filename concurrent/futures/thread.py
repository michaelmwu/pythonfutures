# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ThreadPoolExecutor."""

from __future__ import with_statement
import atexit
import multiprocessing
import threading
import weakref
import sys

from threading import currentThread
from concurrent.futures import _base

try:
    import queue
except ImportError:
    import Queue as queue

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

# Workers are created as daemon threads. This is done to allow the interpreter
# to exit when there are still idle threads in a ThreadPoolExecutor's thread
# pool (i.e. shutdown() was not called). However, allowing workers to die with
# the interpreter has two undesirable properties:
#   - The workers would still be running during interpretor shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads finish.

_thread_references = set()
_shutdown = False

def _python_exit():
    global _shutdown
    _shutdown = True
    for thread_reference in _thread_references:
        thread = thread_reference()
        if thread is not None:
            thread.join()

def _remove_dead_thread_references():
    """Remove inactive threads from _thread_references.

    Should be called periodically to prevent memory leaks in scenarios such as:
    >>> while True:
    ...    t = ThreadPoolExecutor(max_workers=5)
    ...    t.map(int, ['1', '2', '3', '4', '5'])
    """
    for thread_reference in set(_thread_references):
        if thread_reference() is None:
            _thread_references.discard(thread_reference)

atexit.register(_python_exit)

class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException:
            e = sys.exc_info()[1]
            self.future.set_exception(e)
        else:
            self.future.set_result(result)

class ThreadPoolExecutor(_base.Executor):
    def __init__(self, max_workers=None, core_workers=0, keepalive=1):
        """Initializes a new ThreadPoolExecutor instance.

        Args:
            core_workers: The number of persistent threads
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            keepalive: The time for non-core workers to wait for tasks
        """
        _remove_dead_thread_references()

        if max_workers is None:
            max_workers = int(multiprocessing.cpu_count() * 2.5)

        self._max_workers = max(max_workers, core_workers, 1)
        self._core_workers = core_workers
        self._keepalive = keepalive
        self._work_queue = queue.Queue()
        self._threads = set()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()

    def _workitem(self, future, fn, args=None, kwargs=None):
        return _WorkItem(future, fn, args, kwargs)

    def _add_job(self, fn, args=None, kwargs=None, **opts):
        args = args or ()
        kwargs = kwargs or {}

        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = _base.Future()
            w = self._workitem(f, fn, args, kwargs)

            self._work_queue.put(w)
            self._adjust_thread_count()

            return f

    def submit_args(self, fn, args=None, kwargs=None, **opts):
        return self._add_job(fn, args, kwargs, **opts)

    submit_args.__doc__ = _base.Executor.submit_args.__doc__

    def submit(self, fn, *args, **kwargs):
        return self._add_job(fn, args, kwargs)

    submit.__doc__ = _base.Executor.submit.__doc__

    def _worker(self, core):
        block = True
        timeout = None

        if not core:
            block = self._keepalive > 0
            timeout = self._keepalive

        while True:
            try:
                work_item = self._work_queue.get(block, timeout)
            except queue.Empty:
                break

            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown.
            if _shutdown or self._shutdown:
                break
            
            try:
                work_item.run()
            except BaseException:
                _base.LOGGER.critical('Exception in worker', exc_info=True)

        self._threads.remove(currentThread())

    def _adjust_thread_count(self):
        # TODO(bquinlan): Should avoid creating new threads if there are more
        # idle threads than items in the work queue.
        if len(self._threads) < self._max_workers:
            t = threading.Thread(target=self._worker,
                                 args=(len(self._threads) < self._core_workers,))
            t.daemon = True
            t.start()
            self._threads.add(t)
            _thread_references.add(weakref.ref(t))

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True

        for _ in range(len(self._threads)):
            self._work_queue.put(None)

        if wait:
            for t in tuple(self._threads):
                t.join()
    shutdown.__doc__ = _base.Executor.shutdown.__doc__
