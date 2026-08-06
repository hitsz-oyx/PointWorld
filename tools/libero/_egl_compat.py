# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless-rendering compatibility shim for the LIBERO export toolchain.

This module is a project-local shim that makes ``mujoco.OffscreenRenderer``
(and the ``robosuite`` / ``libero`` rendering stacks that wrap it) work on
hosts where the standard EGL driver is not usable -- typically a headless
GPU server where the user has no permission to open ``/dev/dri/renderD*``.

The shim selects one of two backends, depending on the environment:

* ``MUJOCO_GL=egl`` (the default if unset) -- the original NVIDIA EGL
  driver.  We patch the missing ``OpenGL.EGL.EGLDeviceEXT`` ctypes type
  that PyOpenGL doesn't expose by default, and fall back to the default
  EGL display when the ``PLATFORM_DEVICE`` extension isn't usable.

* ``MUJOCO_GL=osmesa`` -- Mesa's CPU-software renderer.  We pre-load
  ``libOSMesa.so.6`` and import ``OpenGL.platform.osmesa`` *before*
  ``mujoco.osmesa`` is imported, otherwise the platform auto-detector
  in PyOpenGL imports ``OpenGL.platform.glx`` and ``mujoco.osmesa``
  blows up with ``ModuleNotFoundError: import of OpenGL halted; None
  in sys.modules``.

Usage
-----
Any script in :mod:`tools.libero` that wants the headless rendering
stack should import this module *first* and call :func:`apply`, e.g.::

    from tools.libero import _egl_compat
    _egl_compat.apply()

    import h5py
    import mujoco
    from libero.libero.envs.env_wrapper import ControlEnv
    ...
"""

from __future__ import annotations

import ctypes
import os


def _load_osmesa() -> None:
    """Pre-load ``libOSMesa.so.6`` and the PyOpenGL OSMesa platform.

    This must be done *before* ``mujoco.osmesa`` is imported, otherwise
    PyOpenGL's platform auto-detector imports ``OpenGL.platform.glx`` and
    ``mujoco.osmesa`` then dies with
    ``ModuleNotFoundError: import of OpenGL halted; None in sys.modules``.
    """
    if os.environ.get("MUJOCO_GL", "").lower() != "osmesa":
        return

    # ``libOSMesa`` must be loaded with ``RTLD_GLOBAL`` so that subsequent
    # OpenGL symbol lookups can find its symbols.  If the library is
    # already loaded by another module, this is a no-op.
    for candidate in ("libOSMesa.so.6", "libOSMesa.so"):
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue

    # Force PyOpenGL to load the OSMesa platform.  This is what
    # ``mujoco.osmesa`` ultimately wants.
    try:
        from OpenGL.platform import osmesa as _osmesa_platform  # noqa: F401
        import OpenGL.GL  # noqa: F401
    except Exception:
        # If we can't load the OSMesa platform here, the downstream
        # ``mujoco.osmesa`` import will fail with a clear error.
        pass


def _patch_opengl_egl() -> None:
    """Install the EGL compatibility patches.

    Only does anything when ``MUJOCO_GL`` is unset, ``egl``, or doesn't
    conflict with ``PYOPENGL_PLATFORM=osmesa`` -- i.e. when the user
    wants the GPU-accelerated path.
    """
    # Honour the user-chosen ``MUJOCO_GL``/``PYOPENGL_PLATFORM`` first;
    # only fall back to EGL if the caller hasn't said anything.
    if "MUJOCO_GL" in os.environ or "PYOPENGL_PLATFORM" in os.environ:
        os.environ.setdefault(
            "PYOPENGL_PLATFORM", os.environ.get("MUJOCO_GL", "egl")
        )
    else:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Only patch if ``OpenGL.EGL`` is importable at all.
    try:
        import OpenGL.EGL as _egl  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - OpenGL may not even be installed
        return

    # (1) Register the missing ``EGLDeviceEXT`` ctypes type.
    if not hasattr(_egl, "EGLDeviceEXT"):
        # In the C header this is ``typedef void *EGLDeviceEXT``; an
        # opaque ``c_void_p`` is the correct C-level stand-in.
        _egl.EGLDeviceEXT = ctypes.c_void_p  # type: ignore[attr-defined]


def _patch_robosuite_egl_fallback() -> None:
    """Patch ``robosuite``'s EGL display init to fall back gracefully.

    When the NVIDIA EGL driver rejects ``PLATFORM_DEVICE`` (which
    happens in many containerized environments) the original
    ``create_initialized_egl_device_display`` returns ``EGL_NO_DISPLAY``
    and the ``EGLGLContext`` constructor aborts.  We wrap it so it
    returns the default EGL display if the device platform is
    unavailable.
    """
    # Only meaningful when the user actually wants the EGL path.
    if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
        return

    try:
        from robosuite.renderers.context import egl_context  # type: ignore
    except Exception:
        return

    if getattr(egl_context, "_pointworld_patched", False):
        return

    import OpenGL.EGL as _egl  # type: ignore[import-not-found]

    _original_create_display = egl_context.create_initialized_egl_device_display

    def _patched_create_initialized_egl_device_display(device_id: int = 0):
        display = _original_create_display(device_id=device_id)
        if display != _egl.EGL_NO_DISPLAY:
            return display

        # Fall back to the default EGL display.
        default_display = _egl.eglGetDisplay(_egl.EGL_DEFAULT_DISPLAY)
        if default_display == _egl.EGL_NO_DISPLAY:
            return default_display

        from OpenGL import error as _gl_error  # type: ignore

        try:
            initialized = _egl.eglInitialize(default_display, None, None)
        except _gl_error.GLError:
            return _egl.EGL_NO_DISPLAY

        if (
            initialized != _egl.EGL_TRUE
            or _egl.eglGetError() != _egl.EGL_SUCCESS
        ):
            return _egl.EGL_NO_DISPLAY
        return default_display

    egl_context.create_initialized_egl_device_display = (
        _patched_create_initialized_egl_device_display
    )
    egl_context._pointworld_patched = True  # type: ignore[attr-defined]


def apply() -> None:
    """Install all headless-rendering compatibility shims.  Idempotent."""
    _load_osmesa()
    _patch_opengl_egl()
    _patch_robosuite_egl_fallback()
