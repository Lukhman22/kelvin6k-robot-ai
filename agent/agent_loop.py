import time
from typing import List, Optional

from agent import config
from agent.rules import Alert, AlertLevel, RuleEngine
from agent.sensors import MachineInterface, NotConnectedMachine, SimulatedSensors
from agent.state import PrinterState
from agent.voice import speak

_LEVEL_PREFIX = {
    AlertLevel.INFO: "Reminder",
    AlertLevel.WARNING: "Warning",
    AlertLevel.CRITICAL: "Critical alert",
}


class MonitoringAgent:
    """
    Process/machine monitoring loop: read sensors and machine telemetry, run the
    rule engine, publish to the shared PrinterState, and speak any alerts.

    Camera analysis runs separately in VisionLoop (main thread) and writes into
    the same PrinterState.
    """

    def __init__(
        self,
        state: Optional[PrinterState] = None,
        rules: Optional[RuleEngine] = None,
        machine: Optional[MachineInterface] = None,
    ):
        self.sensors = SimulatedSensors()
        # Default: no machine link. Channels report as unavailable, never faked.
        self.machine = machine or NotConnectedMachine()
        self.rules = rules or RuleEngine()
        self.state = state or PrinterState()
        self._running = False

    def handle_alerts(self, alerts: List[Alert]) -> None:
        """Prints and speaks alerts. Shared with VisionLoop so camera-derived
        alerts reach the operator through the same path."""
        for alert in alerts:
            print(f"[{_LEVEL_PREFIX[alert.level]}] {alert.message}")
            speak(alert.message)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        print("Machine monitoring started (process sensors are SIMULATED until the machine link is connected).")
        self._running = True
        try:
            while self._running:
                reading = self.sensors.read()
                machine_reading = self.machine.read()

                alerts = self.rules.evaluate(reading, None)
                alerts += self.rules.evaluate_machine(machine_reading)

                self.state.update(reading, None, alerts, machine=machine_reading)
                self.handle_alerts(alerts)

                time.sleep(config.POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
