"""One entry point; defaults to all registered internal research experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from opal.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "research":
        from tools.run_research import main as research_main
        research_main(sys.argv[2:] if len(sys.argv) > 1 else [])
    elif sys.argv[1] == 'next':
        from tools.run_next import main as next_main
        next_main(sys.argv[2:])
    elif sys.argv[1] == 'stage1':
        from tools.run_stage1 import main as stage1_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        stage1_main()
    elif sys.argv[1] == 'ceiling':
        from tools.build_ceiling import main as ceiling_main
        ceiling_main(sys.argv[2:])
    elif sys.argv[1] == 'frontier':
        from tools.build_frontier import main as frontier_main
        frontier_main(sys.argv[2:])
    elif sys.argv[1] == 'architecture':
        from tools.study_architecture import main as architecture_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        architecture_main()
    elif sys.argv[1] == 'improve':
        from tools.improve_residual_model import main as improvement_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        improvement_main()
    else:
        main(sys.argv[1:])
