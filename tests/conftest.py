import os
import sys

# Make the repo root importable so `import model`, `import train`, etc. resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Real validation data (site-local NFS mount; override via env var). The
# NFS-gated tests skip automatically when the mount is absent.
NFS_DATA_DIR = os.environ.get(
    "TARTANIMU_NFS_DATA", "/nfs/data/Processed_Humanoid_Data"
)
import pytest  # noqa: E402  (must follow the sys.path setup above)


def _nfs_available() -> bool:
    return os.path.isdir(NFS_DATA_DIR)


requires_nfs = pytest.mark.skipif(
    not _nfs_available(), reason="NFS data mount not available"
)
