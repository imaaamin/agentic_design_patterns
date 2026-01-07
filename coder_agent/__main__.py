# Entry point for running the package as a module
# Usage: uv run -m coder_agent [--interactive]

import sys
from .code_quality_crew import main, interactive_mode

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

