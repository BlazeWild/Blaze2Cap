#!/usr/bin/env python3
"""Test script to verify NVIDIA EGL setup for MediaPipe"""

import os
import sys

# Set environment variables before importing any EGL-related libraries
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

print("Testing EGL NVIDIA Setup...")
print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
print(f"__EGL_VENDOR_LIBRARY_FILENAMES: {os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', 'NOT SET')}")

try:
    from OpenGL import EGL
    print("\n✓ OpenGL.EGL imported successfully")
    
    # Try to get an EGL display
    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if display == EGL.EGL_NO_DISPLAY:
        print("✗ EGL_NO_DISPLAY returned - cannot get EGL display")
        sys.exit(1)
    
    print(f"✓ Got EGL display: {display}")
    
    # Initialize EGL
    major = EGL.EGLint()
    minor = EGL.EGLint()
    if not EGL.eglInitialize(display, major, minor):
        print("✗ Failed to initialize EGL")
        error = EGL.eglGetError()
        print(f"  EGL Error: 0x{error:04X}")
        sys.exit(1)
    
    print(f"✓ EGL initialized: {major.value}.{minor.value}")
    
    # Get EGL vendor and version
    vendor = EGL.eglQueryString(display, EGL.EGL_VENDOR)
    version = EGL.eglQueryString(display, EGL.EGL_VERSION)
    
    print(f"✓ EGL Vendor: {vendor.decode() if isinstance(vendor, bytes) else vendor}")
    print(f"✓ EGL Version: {version.decode() if isinstance(version, bytes) else version}")
    
    # Terminate EGL
    EGL.eglTerminate(display)
    print("\n✅ NVIDIA EGL is working correctly!")
    
except ImportError as e:
    print(f"\n✗ Failed to import OpenGL.EGL: {e}")
    print("  Installing PyOpenGL might help: pip install PyOpenGL")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ EGL test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
