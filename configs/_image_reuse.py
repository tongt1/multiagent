"""Skip Docker image builds when REUSE_IMAGE_TAG is set.

Import this module as a side-effect in sweep configs to enable
reusing a previously built Docker image tag instead of rebuilding.

Usage in sweep configs:
    import configs._image_reuse  # noqa: F401
"""
import os
import sweep

_tag = os.environ.get("REUSE_IMAGE_TAG")
if _tag:
    sweep.set_reuse_image_tag(_tag)
