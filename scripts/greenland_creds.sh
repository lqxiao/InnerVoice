#!/bin/bash
# Refresh conduit creds into the dedicated `greenland` AWS profile (same step as RufusSafetyRSI/scripts/submit-rsi.sh); default profile untouched.
ada credentials update --account=684288478426 --provider=conduit --role=RufusScienceConduitRole --profile=greenland --once
