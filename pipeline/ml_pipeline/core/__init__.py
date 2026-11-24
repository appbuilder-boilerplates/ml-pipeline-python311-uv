"""Code more than one stage or job of the pipeline uses, one module per concern (logging, ...).

New shared code goes into the module for its concern here, or into a new module (or sub-package) for a concern that
has none yet -- never one function per file, and never inside one stage's own module."""
