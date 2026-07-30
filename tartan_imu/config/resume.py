# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Resume-flag resolution for the training/evaluation CLI.

:func:`configer.build_trainer` only enters its checkpoint-restore branch when
``cfg["train"]["use_pretrain_model"]`` is truthy. A chained job that passes
``--resume_from`` but leaves that flag ``False`` would be silently ignored and
restart from epoch 0 on every resubmit. :func:`resolve_resume` reconciles the
two so an explicit checkpoint path always resumes.
"""

import logging
import os
from typing import Optional


def resolve_resume(cfg: dict, resume_path: Optional[str]) -> bool:
    """Enable pretrained-model loading when a real ``--resume_from`` is given.

    Args:
      cfg: Parsed training config. When a checkpoint file is found, its
        ``["train"]["use_pretrain_model"]`` entry is set to ``True`` in place.
      resume_path: Filesystem path from ``--resume_from``, or ``None``.

    Returns:
      ``True`` if ``resume_path`` points at an existing file (and the flag was
      enabled); ``False`` otherwise.
    """
    if resume_path and os.path.isfile(resume_path):
        cfg["train"]["use_pretrain_model"] = True
        logging.info("Resuming from checkpoint: %s", resume_path)
        return True
    return False
