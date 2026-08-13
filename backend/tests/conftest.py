# conftest.py — keeps pytest from collecting the ad-hoc test_*.py scripts
# in the backend/ root (test_blockchain.py, test_campaign.py, etc.)
# Those are standalone scripts, not pytest tests.
collect_ignore_glob = ["../test_*.py"]
