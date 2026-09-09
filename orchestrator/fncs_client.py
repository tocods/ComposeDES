"""Small ctypes binding for the FNCS C API used by the orchestrator."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List


class FncsClient:
    def __init__(self, library_path: str | None = None) -> None:
        default_library = (
            Path(__file__).resolve().parents[1]
            / "builds"
            / "local"
            / "lib"
            / "libfncs.dylib"
        )
        path = library_path or os.environ.get("FNCS_LIBRARY", str(default_library))
        self._lib = ctypes.CDLL(path)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.fncs_initialize.argtypes = []
        lib.fncs_initialize.restype = None
        lib.fncs_time_request.argtypes = [ctypes.c_ulonglong]
        lib.fncs_time_request.restype = ctypes.c_ulonglong
        lib.fncs_publish.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.fncs_publish.restype = None
        lib.fncs_publish_anon.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.fncs_publish_anon.restype = None
        lib.fncs_finalize.argtypes = []
        lib.fncs_finalize.restype = None
        lib.fncs_get_events_size.argtypes = []
        lib.fncs_get_events_size.restype = ctypes.c_size_t
        lib.fncs_get_events.argtypes = []
        lib.fncs_get_events.restype = ctypes.POINTER(ctypes.c_char_p)
        lib.fncs_get_values_size.argtypes = [ctypes.c_char_p]
        lib.fncs_get_values_size.restype = ctypes.c_size_t
        lib.fncs_get_values.argtypes = [ctypes.c_char_p]
        lib.fncs_get_values.restype = ctypes.POINTER(ctypes.c_char_p)
        lib.fncs_get_value.argtypes = [ctypes.c_char_p]
        lib.fncs_get_value.restype = ctypes.c_void_p
        lib._fncs_free_char_p.argtypes = [ctypes.c_void_p]
        lib._fncs_free_char_p.restype = None
        lib._fncs_free_char_pp.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
        lib._fncs_free_char_pp.restype = None

    def initialize(self) -> None:
        self._lib.fncs_initialize()

    def time_request(self, next_time_ns: int) -> int:
        return int(self._lib.fncs_time_request(next_time_ns))

    def publish(self, key: str, value: str) -> None:
        self._lib.fncs_publish(key.encode("utf-8"), value.encode("utf-8"))

    def publish_anon(self, key: str, value: str) -> None:
        self._lib.fncs_publish_anon(key.encode("utf-8"), value.encode("utf-8"))

    def get_events(self) -> List[str]:
        size = int(self._lib.fncs_get_events_size())
        if size == 0:
            return []
        values = self._lib.fncs_get_events()
        try:
            return [values[index].decode("utf-8") for index in range(size)]
        finally:
            self._lib._fncs_free_char_pp(values, size)

    def get_values(self, key: str) -> List[str]:
        encoded_key = key.encode("utf-8")
        size = int(self._lib.fncs_get_values_size(encoded_key))
        if size == 0:
            return []
        values = self._lib.fncs_get_values(encoded_key)
        try:
            return [values[index].decode("utf-8") for index in range(size)]
        finally:
            self._lib._fncs_free_char_pp(values, size)

    def get_value(self, key: str) -> str:
        value = self._lib.fncs_get_value(key.encode("utf-8"))
        if not value:
            return ""
        try:
            return ctypes.cast(value, ctypes.c_char_p).value.decode("utf-8")
        finally:
            self._lib._fncs_free_char_p(value)

    def finalize(self) -> None:
        self._lib.fncs_finalize()
