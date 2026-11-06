"""Request lifecycle shared by the HART baseline and method subclasses."""

from functools import wraps
from inspect import signature


def hart_inference_session(infer):
    """Release request caches even when preparation or generation fails."""
    infer_signature = signature(infer)

    @wraps(infer)
    def wrapped(self, *args, **kwargs):
        # A disabled method can delegate to the already guarded baseline.
        if getattr(self, "_hart_inference_active", False):
            return infer(self, *args, **kwargs)
        arguments = infer_signature.bind(self, *args, **kwargs)
        arguments.apply_defaults()
        self.reset_inference_state()
        self.last_inference_metadata = {}
        self._hart_inference_active = True
        if hasattr(self, "_hart_request_cache"):
            self._hart_request_cache["active_inference"] = True
        try:
            self._validate_inference_request(arguments.arguments.get("num_maskgit_iters", 1))
            self.prepare_for_inference()
            return infer(self, *args, **kwargs)
        finally:
            try:
                self.reset_inference_state()
            finally:
                self._hart_inference_active = False

    return wrapped
