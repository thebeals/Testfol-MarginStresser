"""Shared magic strings used across the codebase."""


class Freq:
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    YEARLY = "Yearly"


class RebalMode:
    NONE = "None"
    STANDARD = "Standard"
    CUSTOM = "Custom"
    THRESHOLD = "Threshold"
    THRESHOLD_CALENDAR = "Threshold+Calendar"


class DcaMode:
    PROPORTIONAL = "Proportional"
    SINGLE_ASSET = "Single Asset"


class Tickers:
    NDXMEGASIM = "NDXMEGASIM"
    NDXMEGA2SIM = "NDXMEGA2SIM"
    QQUPSIM = "QQUPSIM"
    NDX30SIM = "NDX30SIM"
    QBIG = "QBIG"
