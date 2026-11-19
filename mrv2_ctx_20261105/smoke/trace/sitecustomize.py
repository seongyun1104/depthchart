# Put this directory on PYTHONPATH. With MRV2_TRACE_DIR set, each process records
# the scheduler's K decisions and the MRv2 graph managers' dispatch, capture and
# replay calls (see mrv2_trace.py).
import mrv2_trace

mrv2_trace.activate()
