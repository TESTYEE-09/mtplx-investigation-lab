# SPDX-License-Identifier: Apache-2.0
"""Vendored vLLM-Metal kernel subset used by MTPLX.

Only the standalone Metal paged-attention extension is bundled here.  MTPLX
does not vendor or depend on the vLLM serving stack.
"""

__all__ = ["metal"]
