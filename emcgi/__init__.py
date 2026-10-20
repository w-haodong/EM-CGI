"""EM-CGI core implementation."""
from .config import model_config
from .models import EMCGI

__all__ = ["EMCGI", "model_config"]
