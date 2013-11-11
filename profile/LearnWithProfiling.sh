#! /bin/bash
# LearnWithProfiling.sh
#USAGE
# LearnWithProfiling.sh [dataName] [allocModelName] [obsModelName] [algName] [opts]

# Prepare directory for output
reportDir=profilerReports/
mkdir -p $reportDir

# Remove any previous decorations from bnpy code
# Just in case last profile run didn't clean up properly
# Harmless if not needed
python undecorate_for_profiling.py

# Decorate bnpy code for profiling
python decorate_for_profiling.py

pushd ..

# Profile the execution of Learn.py
# Write results to a lprof file.
python profile/kernprof.py --line-by-line Learn.py $*

# Convert lprof file into a plain-text report
# Called pyprofile.txt
python -m line_profiler Learn.py.lprof > $reportDir/report.txt
rm Learn.py.lprof

popd

# Remove functions that didn't get any runtime from report
python scrub_profile_report.py

# Remove decorations from bnpy code
python undecorate_for_profiling.py
echo "Wrote final report to: $reportDir/report.txt"
