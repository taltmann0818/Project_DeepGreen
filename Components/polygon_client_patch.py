import certifi
import json
import logging
import urllib3
from urllib3.util.retry import Retry
from polygon.rest.base import BaseClient, logger, version_number
from polygon.exceptions import AuthError


def patch_polygon_client(max_pool_size: int = 10) -> None:
    """Monkeypatch polygon BaseClient to use a larger per-host connection pool."""

    def new_init(
        self,
        api_key: str = 'XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2',
        connect_timeout: float = 10.0,
        read_timeout: float = 10.0,
        num_pools: int = 10,
        retries: int = 3,
        base: str = "https://api.polygon.io",
        verbose: bool = False,
        trace: bool = False,
        custom_json=None,
    ):
        if api_key is None:
            raise AuthError(
                "Must specify env var POLYGON_API_KEY or pass api_key in constructor"
            )

        self.API_KEY = api_key
        self.BASE = base

        self.headers = {
            "Authorization": "Bearer " + self.API_KEY,
            "Accept-Encoding": "gzip",
            "User-Agent": f"Polygon.io PythonClient/{version_number}",
        }

        self.retries = retries

        retry_strategy = Retry(
            total=self.retries,
            status_forcelist=[413, 429, 499, 500, 502, 503, 504],
            backoff_factor=0.1,
        )

        self.client = urllib3.PoolManager(
            num_pools=num_pools,
            maxsize=max_pool_size,
            headers=self.headers,
            ca_certs=certifi.where(),
            cert_reqs="CERT_REQUIRED",
            retries=retry_strategy,
        )

        self.timeout = urllib3.Timeout(connect=connect_timeout, read=read_timeout)

        if verbose:
            logger.setLevel(logging.DEBUG)
        self.trace = trace
        if custom_json:
            self.json = custom_json
        else:
            self.json = json

    BaseClient.__init__ = new_init
