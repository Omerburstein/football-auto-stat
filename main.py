"""Run detection + identity tracking for a match: `py -3.13 main.py matches/<game>/match.yaml`.

See tracking/pipeline.py for options and matches/example/match.yaml for the inputs.
"""
from tracking.pipeline import main

if __name__ == "__main__":
    main()
