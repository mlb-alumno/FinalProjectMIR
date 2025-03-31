import glob
import os
from decimal import ROUND_HALF_UP, Decimal

import jams
import numpy as np

# the pumpp library hard-codes a slash in cqt/mag, newer versions of keras don't allow the use of / in keys so this functions were written to solve this   

def remove_slashes_from_npz_keys(folder):
    """
    Remove slashes from keys in .npz files within the specified folder.

    Args:
        folder (str): Path to the folder containing .npz files.
    """
    for fname in glob.glob(os.path.join(folder, "*.npz")):
        data = np.load(fname)
        new_data = {}
        for k in data.keys():
            new_k = k.replace("/", "_")  # e.g. cqt/mag -> cqt_mag
            new_data[new_k] = data[k]
        data.close()

        # Overwrites the same file, but now with slash-free keys
        np.savez(fname, **new_data)


def rename_slashes_in_op_fields(op):
    """
    In-place rename of all slash-laden keys inside op.fields
    so that the new keys replace '/' with '_'.
    """
    # Must check that op.fields is actually a mutable dict
    if hasattr(op, "fields") and isinstance(op.fields, dict):
        new_dict = {}
        for old_key, old_val in op.fields.items():
            new_key = old_key.replace("/", "_")  # e.g. 'cqt/mag' -> 'cqt_mag'
            new_dict[new_key] = old_val
        # Now overwrites op.fields with slash-free keys
        op.fields = new_dict


def rename_slashes_in_pump_ops_list(pump):
    """
    pump.ops is a list of the same operators, rename slash-based keys in each.
    """
    for op in pump.ops:
        rename_slashes_in_op_fields(op)


def rename_slashes_in_pump_opmap(pump):
    """
    Go through pump.opmap, rename slash-based keys in each operator's fields.
    """
    for op_name, op in pump.opmap.items():
        rename_slashes_in_op_fields(op)


def round_observation_times(annotation, precision=5, snap_tol=1e-6):
    """
    Creates new Observation objects with times and durations rounded using Decimal
    arithmetic, then force them to be consecutive by snapping boundaries that are
    within snap_tol.

    Args:
        annotation (JAMS Annotation): A JAMS-style chord annotation with Observation objects.
        precision (int): Decimal places to round to.
        snap_tol (float): Tolerance under which boundaries are forced equal.

    Returns:
        JAMS Annotation: The adjusted annotation.
    """

    # Define quantizer string and quant
    quant_str = "1." + "0" * precision
    quant = Decimal(quant_str)

    # First pass: convert observation times/durations to Decimal.
    obs_list = []
    for obs in annotation.data:
        rt = Decimal(str(obs.time)).quantize(quant, rounding=ROUND_HALF_UP)
        rd = Decimal(str(obs.duration)).quantize(quant, rounding=ROUND_HALF_UP)
        obs_list.append((rt, rd, obs.value, obs.confidence))

    # Sort by start time.
    obs_list.sort(key=lambda tup: tup[0])

    # Second pass: force consecutive intervals.
    fixed = []
    # Start with the first observation.
    prev_start, prev_dur, val, conf = obs_list[0]
    prev_end = prev_start + prev_dur
    fixed.append((prev_start, prev_dur, val, conf))

    for current in obs_list[1:]:
        current_start, current_dur, val, conf = current
        # Force the current observation to start at prev_end
        new_start = prev_end
        # Calculate original end of current observation.
        current_end = current_start + current_dur
        # New duration is calculated as difference.
        new_dur = current_end - new_start
        if new_dur < Decimal("0"):
            new_dur = Decimal("0")
        fixed.append((new_start, new_dur, val, conf))
        prev_end = new_start + new_dur  # update end

    # Convert fixed intervals back to floats with snapping.
    fixed_obs = []
    # Build the new observations, and whenever the gap is below snap_tol, snap them.
    prev_end_float = None
    for start, dur, val, conf in fixed:
        start_float = float(start)
        dur_float = float(dur)
        end_float = start_float + dur_float
        if (
            prev_end_float is not None
            and abs(start_float - prev_end_float) < snap_tol
        ):
            # snap start exactly to previous end.
            start_float = prev_end_float
            # Adjust duration based on the original end.
            end_float = float(start + dur)
            dur_float = max(0, end_float - start_float)
        obs_new = jams.Observation(
            time=start_float,
            duration=dur_float,
            value=val,
            confidence=conf,
        )
        fixed_obs.append(obs_new)
        prev_end_float = start_float + dur_float

    annotation.data = fixed_obs
    return annotation
