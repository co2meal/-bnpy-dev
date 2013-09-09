#! /bin/bash

python undecorate_for_profiling.py # Harmless if unnecessary
python decorate_for_profiling.py

pushd ..

python profile/kernprof.py --line-by-line Learn.py $*
python -m line_profiler Learn.py.lprof > profile/profiles/pyprofile.txt

rm Learn.py.lprof

popd

# Remove functions that didn't get any runtime from report
python scrub_profile_report.py

python undecorate_for_profiling.py

echo "Wrote final report to: profiles/pyprofile.txt"


