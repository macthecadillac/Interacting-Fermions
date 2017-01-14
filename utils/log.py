import functools
import logging
import os
import tempfile

logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(os.path.join(
                              tempfile.gettempdir(), "logged.log"))
logger.addHandler(handler)


def logged(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        log = "called: " + function.__name__ + "("
        log += ", ".join(["{0!r}".format(a) for a in args] +
                         ["{0!s}={1!r}".format(k, v)
                          for k, v in kwargs.items()])
        result = exception = None
        try:
            result = function(*args, **kwargs)
            return result
        except Exception as err:
            exception = err
        finally:
            log += ((") -> " + str(result)) if exception is None
                    else ") {0}: {1}".format(type(exception),
                                             exception))
            logger.debug(log)
            if exception is not None:
                raise exception
    return wrapper
