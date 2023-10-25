from autonmt.toolkits.autonmt import AutonmtTranslator

try:
    from autonmt.toolkits.fairseq import FairseqTranslator
except ImportError as e:
    print("WARNING: Fairseq toolkit could not be loaded. FairseqTranslator will not be available.")
except Exception as e:
    print("WARNING: Fairseq toolkit could not be loaded. FairseqTranslator will not be available.")
    print(e)
