"""
CLI Runner for Track A: Public Diverse OOF Stacking
===================================================
Discovers public models in data/public_oof/ or /kaggle/input/,
performs 10-fold L2-regularized logistic stacking with internal 3 fast GBDTs,
applies monotonic rank transformation, and updates champion submission.
"""

from src.stack_public import main

if __name__ == "__main__":
    main()
