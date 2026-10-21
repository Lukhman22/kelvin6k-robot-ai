import random
import time
from dataclasses import dataclass
from typing import Optional

# Data-source labels. Every value the agent reports carries one of these so the
# operator (and the voice assistant) always knows whether a number is real.
SIMULATED = "SIMULATED"
NOT_CONNECTED = "NOT_CONNECTED"

_NO_MACHINE_LINK = (
    "No machine communication link is connected. Connect the printer/robot "
    "controller (e.g. serial, Modbus, OPC-UA, or its REST API) and implement a "
    "MachineInterface subclass to supply these values."
)


@dataclass
class SensorReading:
    """Process values. Currently produced by the simulator -- see `source`."""

    timestamp: float
    nozzle_temp_c: float
    bed_temp_c: float
    print_height_mm: float
    expected_height_mm: float
    material_flow_percent: float
    machine_state: str  # "printing" | "error"
    source: str = SIMULATED


@dataclass
class MachineReading:
    """
    Real machine telemetry channels.

    Every field defaults to None, meaning "this channel is not connected".
    Nothing here is ever simulated -- an unconnected channel is reported as
    unavailable rather than filled with a plausible-looking number.
    """

    pump_running: Optional[bool] = None
    pump_speed_rpm: Optional[float] = None
    accelerator_running: Optional[bool] = None
    accelerator_speed_rpm: Optional[float] = None
    print_speed_mm_s: Optional[float] = None
    feed_speed_mm_s: Optional[float] = None
    pressure_bar: Optional[float] = None
    motor_status: Optional[str] = None
    error_state: Optional[str] = None
    # Machine-side print job info -- the only trustworthy source of absolute
    # layer numbering and progress (the camera cannot know these).
    current_layer_number: Optional[int] = None
    total_layers: Optional[int] = None
    source: str = NOT_CONNECTED
    unavailable_reason: str = _NO_MACHINE_LINK


class SimulatedSensors:
    """
    SIMULATED process-value source -- NOT real machine data.

    Produces plausible, slowly-drifting readings so the rule engine and voice
    pipeline can be exercised without hardware. Every reading it returns is
    labelled `source=SIMULATED` and is surfaced to the operator as such.

    Replace with a real implementation (same `read()` interface and return type)
    once the machine link exists.
    """

    def __init__(self):
        self._actual_height = 0.0
        self._expected_height = 0.0
        self._nozzle_temp = 210.0
        self._bed_temp = 60.0
        self._flow = 100.0

    def read(self) -> SensorReading:
        nominal_step = 0.1
        self._expected_height += nominal_step
        self._actual_height += nominal_step * random.uniform(0.8, 1.05)

        self._nozzle_temp += random.uniform(-2, 2)
        self._bed_temp += random.uniform(-1, 1)
        self._flow += random.uniform(-3, 2)
        self._flow = max(0.0, min(100.0, self._flow))

        # Occasionally simulate a problem so alerts have something to fire on.
        if random.random() < 0.05:
            self._flow -= random.uniform(20, 40)
        if random.random() < 0.03:
            self._nozzle_temp += random.uniform(15, 30)

        machine_state = "error" if random.random() < 0.02 else "printing"

        return SensorReading(
            timestamp=time.time(),
            nozzle_temp_c=round(self._nozzle_temp, 1),
            bed_temp_c=round(self._bed_temp, 1),
            print_height_mm=round(self._actual_height, 2),
            expected_height_mm=round(self._expected_height, 2),
            material_flow_percent=round(max(0.0, self._flow), 1),
            machine_state=machine_state,
            source=SIMULATED,
        )


class MachineInterface:
    """
    Base class for a real machine link (pump, accelerator, drives, job info).

    Implement `read()` against your controller and pass an instance to
    MonitoringAgent to replace the not-connected default. Do NOT return
    invented values for channels your hardware doesn't expose -- leave them
    None so the agent reports them as unavailable.
    """

    def read(self) -> MachineReading:
        raise NotImplementedError


class NotConnectedMachine(MachineInterface):
    """Default: no machine link. Every channel reports as unavailable."""

    def read(self) -> MachineReading:
        return MachineReading()
