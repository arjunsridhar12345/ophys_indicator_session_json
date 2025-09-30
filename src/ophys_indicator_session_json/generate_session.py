from aind_data_schema.components.stimulus import OptoStimulation
from aind_data_schema.core.session import (
    StimulusEpoch,
    Stream,
    DetectorConfig,
    FiberConnectionConfig,
    LightEmittingDiodeConfig,
    Session
)
from aind_data_schema_models.modalities import Modality
from pathlib import Path
import argparse
import json
import pandas as pd
import re
from datetime import datetime


sample_path = Path(r"\\allen\aind\scratch\KentaHagihara_InternalTransfer\428-9-C_transfer\807928_LH_screening_20hz_2s_stim_15ms_pw_10_trials_1mW\2025_08_01\fib")

"""Fiber Photometry job settings configuration."""

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class FiberData(BaseModel):
    """
    Intermediate data model for fiber photometry data.

    This model holds the extracted and processed data before final
    transformation into a Session object. It serves as a structured
    intermediate representation of the fiber photometry session data.
    """

    job_settings_name: Literal["FiberPhotometry"] = Field(
        default="FiberPhotometry", description="Name of the job settings type"
    )
    experimenter_full_name: List[str] = Field(..., description="List of experimenter names")
    session_start_time: Optional[datetime] = Field(None, description="Start time of the session")
    session_end_time: Optional[datetime] = Field(None, description="End time of the session")
    subject_id: str = Field(..., description="Subject identifier")
    rig_id: str = Field(..., description="Identifier for the experimental rig")
    mouse_platform_name: Optional[str] = Field(..., description="Name of the mouse platform used")
    active_mouse_platform: Optional[bool] = Field(
        ..., description="Whether the mouse platform was active during the session"
    )
    data_streams: List[dict] = Field(default_factory=list, description="List of data stream configurations")
    session_type: str = Field(default="FIB", description="Type of session")
    iacuc_protocol: str = Field(..., description="IACUC protocol identifier")
    notes: str = Field(..., description="Session notes")
    anaesthesia: Optional[str] = Field(None, description="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(None, description="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(None, description="Animal weight before session")
    protocol_id: List[str] = Field(default_factory=list, description="List of protocol identifiers")
    data_directory: Optional[Union[str, Path]] = Field(
        None, description="Path to data directory containing fiber photometry files"
    )
    local_timezone: str = Field(default="America/Los_Angeles", description="Timezone for the session")
    output_directory: Optional[Union[str, Path]] = Field(None, description="Output directory for generated files")
    output_filename: str = Field(default="session_fip.json", description="Name of output file")


class FiberJobSettings(BaseModel):
    """Data to be entered by the user."""

    # Field can be used to switch between different acquisition etl jobs
    job_settings_name: Literal["FiberPhotometry"] = "FiberPhotometry"

    # Required fields
    subject_id: str = Field(default="FIB", description="Subject id")
    rig_id: str = Field(default="FIB", description="Rig id")
    iacuc_protocol: str = Field(default="FIB", description="Type of session")
    notes: str = Field(default="FIB", description="Type of session")

    # Session metadata
    experimenter_full_name: List[str] = Field(default_factory=list, description="List of experimenter names")
    session_type: str = Field(default="FIB", description="Type of session")
    mouse_platform_name: Optional[str] = Field(default=None, description="Name of the mouse platform used")
    active_mouse_platform: bool = Field(
        default=False, description="Whether the mouse platform was active during the session"
    )

    # Optional session details
    anaesthesia: Optional[str] = Field(default=None, description="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(default=None, description="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(default=None, description="Animal weight before session")

    # Data configuration
    data_streams: List[dict] = Field(default_factory=list, description="List of data stream configurations")
    protocol_id: List[str] = Field(default_factory=list, description="List of protocol identifiers")

    # File paths
    data_directory: Optional[Union[str, Path]] = Field(
        default=None, description="Path to data directory containing fiber photometry files"
    )
    output_directory: Optional[Union[str, Path]] = Field(
        default=None, description="Output directory for generated files"
    )
    output_filename: str = Field(default="session.json", description="Name of output file")

    # Timing configuration
    local_timezone: str = Field(default="America/Los_Angeles", description="Timezone for the session")


REGEX_DATE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
REGEX_MOUSE_ID = r"mouse_\w+"


class FiberPhotometryExtractor:
    """Extractor for Fiber Photometry metadata from legacy data files."""

    def __init__(self, job_settings: FiberJobSettings):
        """Initialize the Fiber Photometry extractor with job settings."""
        self.job_settings = job_settings

    @classmethod
    def from_args(cls, args: List[str]) -> "FiberPhotometryExtractor":
        """Create FiberPhotometryExtractor from command line arguments.

        Parameters
        ----------
        args : List[str]
            Command line arguments

        Returns
        -------
        FiberPhotometryExtractor
            Configured extractor instance
        """
        parser = argparse.ArgumentParser(description="Fiber Photometry ETL Job Settings")

        # Required arguments
        parser.add_argument("--subject_id", required=True, help="Subject identifier")
        parser.add_argument("--rig_id", required=True, help="Rig identifier")
        parser.add_argument("--iacuc_protocol", required=True, help="IACUC protocol")
        parser.add_argument("--notes", required=True, help="Session notes")
        parser.add_argument("--data_directory", required=True, help="Data directory path")

        # Optional arguments
        parser.add_argument(
            "--experimenter_full_name",
            nargs="+",
            default=[],
            help="Experimenter names",
        )
        parser.add_argument("--session_type", default="FIB", help="Session type")
        parser.add_argument("--mouse_platform_name", help="Mouse platform name")
        parser.add_argument(
            "--active_mouse_platform",
            action="store_true",
            help="Mouse platform active",
        )
        parser.add_argument("--anaesthesia", help="Anaesthesia used")
        parser.add_argument(
            "--animal_weight_post",
            type=float,
            help="Animal weight post session",
        )
        parser.add_argument(
            "--animal_weight_prior",
            type=float,
            help="Animal weight prior to session",
        )
        parser.add_argument(
            "--local_timezone",
            default="America/Los_Angeles",
            help="Local timezone",
        )
        parser.add_argument("--output_directory", help="Output directory")
        parser.add_argument(
            "--output_filename",
            default="session_fip.json",
            help="Output filename",
        )
        parser.add_argument("--data_streams", help="JSON string of data streams configuration")

        parsed_args = parser.parse_args(args)

        # Parse data_streams if provided as JSON string
        data_streams = []
        if parsed_args.data_streams:
            data_streams = json.loads(parsed_args.data_streams)

        job_settings = FiberJobSettings(
            subject_id=parsed_args.subject_id,
            rig_id=parsed_args.rig_id,
            iacuc_protocol=parsed_args.iacuc_protocol,
            notes=parsed_args.notes,
            data_directory=parsed_args.data_directory,
            experimenter_full_name=parsed_args.experimenter_full_name,
            session_type=parsed_args.session_type,
            mouse_platform_name=parsed_args.mouse_platform_name,
            active_mouse_platform=parsed_args.active_mouse_platform,
            anaesthesia=parsed_args.anaesthesia,
            animal_weight_post=parsed_args.animal_weight_post,
            animal_weight_prior=parsed_args.animal_weight_prior,
            local_timezone=parsed_args.local_timezone,
            output_directory=parsed_args.output_directory,
            output_filename=parsed_args.output_filename,
            data_streams=data_streams,
        )

        return cls(job_settings)

    def extract(self) -> dict:
        """Run extraction process"""

        file_metadata = self._extract_metadata_from_data_files()

        # Create the fiber data model
        fiber_data = FiberData(**file_metadata)

        return fiber_data.model_dump()

    def _extract_metadata_from_data_files(self) -> dict:
        """
        Extracts metadata from the fiber photometry data files.

        Returns
        -------
        Dict
            Dictionary containing metadata from
            the data files for the current acquisition.
        """
        # Convert input_source to Path - handle various input types
        if isinstance(self.job_settings.data_directory, (str, Path)):
            data_dir = Path(self.job_settings.data_directory)
        else:
            raise ValueError("data_directory must be a valid path")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Find FIP data files
        data_files = list(data_dir.glob("FIP_Data*.csv"))
        if not data_files:
            # Try alternative patterns
            data_files = list(data_dir.glob("*Signal*.csv")) + list(data_dir.glob("*Stim*.csv"))

        if not data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        # Extract session timing
        start_time, end_time = self._extract_session_timing(data_files)

        metadata_dict = {
            "job_settings_name": "FiberPhotometry",
            "experimenter_full_name": self.job_settings.experimenter_full_name,
            "session_start_time": start_time,
            "session_end_time": end_time,
            "subject_id": self.job_settings.subject_id,
            "rig_id": self.job_settings.rig_id,
            "mouse_platform_name": self.job_settings.mouse_platform_name,
            "active_mouse_platform": self.job_settings.active_mouse_platform,
            "data_streams": self.job_settings.data_streams,
            "session_type": self.job_settings.session_type,
            "iacuc_protocol": self.job_settings.iacuc_protocol,
            "notes": self.job_settings.notes,
            "anaesthesia": self.job_settings.anaesthesia,
            "animal_weight_post": self.job_settings.animal_weight_post,
            "animal_weight_prior": self.job_settings.animal_weight_prior,
            "protocol_id": self.job_settings.protocol_id,
            "data_directory": str(data_dir),
            "data_files": [str(f) for f in data_files],
            "local_timezone": self.job_settings.local_timezone,
            "output_directory": self.job_settings.output_directory,
            "output_filename": self.job_settings.output_filename,
        }

        return metadata_dict

    def _extract_session_start_time(self, data_files: List[Path]) -> Optional[datetime]:
        """
        Extract session start time from filenames using regex.

        Parameters
        ----------
        data_files : List[Path]
            List of data file paths to extract the start time from.

        Returns
        -------
        Optional[datetime]
            Extracted session start time or None if not found.
        """
        for file_path in data_files:
            match = re.search(REGEX_DATE, file_path.name)
            if match:
                try:
                    return datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")
                except Exception:
                    continue
        raise ValueError("Could not extract valid timestamp from filenames")

    def _extract_session_end_time(self, data_files: List[Path]) -> Optional[datetime]:
        """
        Extract session end time from CSV content if possible.

        Parameters
        ----------
        data_files : List[Path]
            List of data file paths to extract the end time from.

        Returns
        -------
        Optional[datetime]
            Extracted session end time or None if not found.
        """
        for file_path in data_files:
            try:
                timestamp_cols = [col for col in df.columns if "time" in col.lower() or "timestamp" in col.lower()]
                if timestamp_cols:
                    timestamps = pd.to_datetime(df[timestamp_cols[0]])
                    return timestamps.max().to_pydatetime()
            except Exception:
                continue
        return None

    def _extract_session_timing(self, data_files: List[Path]) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract session start and end times from data files."""
        if not data_files:
            return None, None

        try:
            # Read the first data file to get timing information
            signal_file = data_files[0]
            df_signal = pd.read_csv(signal_file)

            # Try to find timestamp column
            timestamp_cols = [col for col in df_signal.columns if "ts" in col.lower()]
            if timestamp_cols:
                start_time = datetime.strptime(signal_file.stem, "Signal_%Y-%m-%dT%H_%M_%S")

                timestamps_signal = df_signal[timestamp_cols[0]]

                start_time = start_time + pd.to_timedelta(timestamps_signal.min())
                end_time = None
                return start_time, end_time
            else:
                # Try to extract from filename if timestamp column not found
                return self._extract_timing_from_filename(signal_file)

        except Exception:
            # Fallback to filename extraction
            return self._extract_timing_from_filename(data_files[0])

    def _extract_timing_from_filename(self, file_path: Path) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract timing information from filename using regex."""
        try:
            # Try to match date pattern in filename or parent directory
            for path_part in [file_path.name, file_path.parent.name]:
                date_match = re.search(REGEX_DATE, path_part)
                if date_match:
                    start_time = datetime.strptime(date_match.group(), "%Y-%m-%d_%H-%M-%S")
                    return start_time, None
            return None, None
        except Exception:
            return None, None

class OptoJobSettings(BaseModel):
    """Parameters for extracting from raw data."""

    data_directory: str = Field(
        ...,
        description="Path to data directory",
    )

    # Optogenetics parameters
    stimulus_name: str = Field(default="OptoStim", description="Stimulus name")
    pulse_shape: str = Field(default="Square", title="Pulse shape")
    pulse_frequency: List[float] = Field(default=[10], title="Pulse frequency (Hz)")
    number_pulse_trains: List[int] = Field(default=[30], title="Number of pulse trains")
    pulse_width: List[int] = Field(default=[5], title="Pulse width (ms)")

    pulse_train_duration: List[float] = Field(
        default=[2], title="Pulse train duration (s)"
    )
    fixed_pulse_train_interval: bool = Field(
        default=True, title="Fixed pulse train interval"
    )
    pulse_train_interval: Optional[float] = Field(
        default=28,
        title="Pulse train interval (s)",
        description="Time between pulse trains",
    )
    baseline_duration: float = Field(
        default=120,
        title="Baseline duration (s)",
        description="Duration of baseline recording prior to first pulse train",
    )

    # Stimulus epoch laser configs
    wavelength: int = Field(default=100, title="Wavelength (nm)")
    power: float = Field(default=10, title="Excitation power")

class OptoModel(BaseModel):
    """Ophys Indicator Benchmark model for extracting metadata."""

    opto_metadata: dict = Field(
        ...,
        title="Optogenetics metadata",
        description="Metadata for Optogenetics",
    )

    stimulus_epochs: dict = Field(
        ...,
        title="Optogenetics Stimulus Epochs",
        description="Optogenetics stimulus epoch information",
    )


class OphysIndicatorBenchmarkModel(BaseModel):
    """Intermediate data structure"""

    opto_data: OptoModel
    fiber_data: FiberData


class OphysIndicatorBenchMarkExtractor:
    """Extractor for Ophys Benchmark Opto Metadata."""

    def __init__(self, job_settings: Union[str, OptoJobSettings], fiber_settings: FiberJobSettings):
        """Initialize the Ophys Benchmark extractor with job settings."""
        if isinstance(job_settings, str):
            if Path(job_settings).exists():
                with open(job_settings, "r") as f:
                    jobs_settings_params = json.load(f)
                    job_settings = json.dumps(jobs_settings_params)

            self.job_settings = OptoJobSettings.model_validate_json(job_settings)
        else:
            self.job_settings = job_settings
        
        self.fiber_data = FiberPhotometryExtractor(fiber_settings)

    def extract(self) -> OphysIndicatorBenchmarkModel:
        """Run extraction process"""
        opto_params = self._extract_opto_parameters()
        stimulus_epochs = self._extract_stimulus_epochs()
        opto_model = OptoModel(
            opto_metadata=opto_params, stimulus_epochs=stimulus_epochs
        )

        fiber_metadata = FiberData(**self.fiber_data.extract())
        return OphysIndicatorBenchmarkModel(opto_data=opto_model, fiber_data=fiber_metadata)

    def _extract_opto_parameters(self) -> dict:
        """Returns opto parameters"""
        return {
            "stimulus_name": self.job_settings.stimulus_name,
            "pulse_shape": self.job_settings.pulse_shape,
            "pulse_frequency": self.job_settings.pulse_frequency,
            "number_pulse_trains": self.job_settings.number_pulse_trains,
            "pulse_width": self.job_settings.pulse_width,
            "pulse_train_duration": self.job_settings.pulse_train_duration,
            "fixed_pulse_train_interval": self.job_settings.fixed_pulse_train_interval,
            "pulse_train_interval": self.job_settings.pulse_train_interval,
            "baseline_duration": self.job_settings.baseline_duration,
        }

    def _extract_stimulus_epochs(self) -> dict:
        """Extracts stimulus epoch information"""
        if isinstance(self.job_settings.data_directory, str):
            self.job_settings.data_directory = Path(self.job_settings.data_directory)

        stim_csv_path = tuple(
            self.job_settings.data_directory.glob("Stim*.csv")
        )
        if not stim_csv_path:
            raise FileNotFoundError("No stim csv found. Check data")

        stim_df = pd.read_csv(stim_csv_path[0])
        filename = stim_csv_path[0].stem

        # Parse directly with the format
        start_time = datetime.strptime(filename, "Stim_%Y-%m-%dT%H_%M_%S")
        start_time = start_time + pd.to_timedelta(self.job_settings.baseline_duration, unit="s")
        # Compute end time

        
        end_time = start_time + pd.to_timedelta(
            (self.job_settings.pulse_train_duration[0] + self.job_settings.pulse_train_interval) * self.job_settings.number_pulse_trains[0], unit="s"
        )
        end_time = end_time.isoformat()

        return {
            "stimulus_start_time": start_time.isoformat(),
            "stimulus_end_time": end_time,
            "stimulus_name": "OptoStim",
            "stimulus_modalities": ["Optogenetics"],
            "configurations": {
                "wavelength": self.job_settings.wavelength,
                "power": self.job_settings.power,
            },
        }

class JobSettings(BaseSettings, cli_parse_args=True):
    fiber: FiberJobSettings = Field(default_factory=FiberJobSettings)
    opto: OptoJobSettings = Field(default_factory=OptoJobSettings)
    fiber_streams_path: str = Field(default_factory=str)
    opto_params_path: str = Field(default_factory=str)

if __name__ == "__main__":
    settings = JobSettings()

    settings.fiber.data_directory = settings.opto.data_directory

    with open(settings.fiber_streams_path, "r") as f:
        fiber_streams = json.load(f)
    
    with open(settings.opto_params_path, "r") as f:
        opto_params = json.load(f)
    
    opto_params["data_directory"] = settings.opto.data_directory
    settings.fiber.data_streams = fiber_streams
    settings.opto = OptoJobSettings(**opto_params)

    opto_model = OphysIndicatorBenchMarkExtractor(settings.opto, settings.fiber).extract()
    opto_model.fiber_data.session_end_time = opto_model.opto_data.stimulus_epochs["stimulus_end_time"]
    print(opto_model.model_dump_json())

    stimulus_epoch = StimulusEpoch(
        stimulus_start_time=opto_model.opto_data.stimulus_epochs["stimulus_start_time"],
        stimulus_end_time=opto_model.opto_data.stimulus_epochs["stimulus_end_time"],
        stimulus_name=opto_model.opto_data.opto_metadata["stimulus_name"],
        stimulus_modalities=opto_model.opto_data.stimulus_epochs["stimulus_modalities"],
        stimulus_parameters=[
            OptoStimulation(
                stimulus_name=opto_model.opto_data.opto_metadata["stimulus_name"],
                pulse_shape=opto_model.opto_data.opto_metadata["pulse_shape"],
                pulse_frequency=opto_model.opto_data.opto_metadata["pulse_frequency"],
                number_pulse_trains=opto_model.opto_data.opto_metadata["number_pulse_trains"],
                pulse_width=opto_model.opto_data.opto_metadata["pulse_width"],
                pulse_train_duration=opto_model.opto_data.opto_metadata["pulse_train_duration"],
                fixed_pulse_train_interval=opto_model.opto_data.opto_metadata["fixed_pulse_train_interval"],
                pulse_train_interval=opto_model.opto_data.opto_metadata["pulse_train_interval"],
                baseline_duration=opto_model.opto_data.opto_metadata["baseline_duration"]
            )
        ]
    )


    data_stream = Stream(
        stream_start_time=opto_model.fiber_data.session_start_time,
        stream_end_time=opto_model.fiber_data.session_end_time,
        light_sources=[
            LightEmittingDiodeConfig(**ls)
            for ls in opto_model.fiber_data.data_streams[0]["light_sources"]
        ],
        stream_modalities=[Modality.FIB],
        detectors=[
            DetectorConfig(**d) for d in opto_model.fiber_data.data_streams[0]["detectors"]
        ],
        fiber_connections=[
            FiberConnectionConfig(**fc) for fc in opto_model.fiber_data.data_streams[0]["fiber_connections"]
        ],
        
    )
    # TODO: Add light source config for LaserConfig to StimulusEpoch

    session = Session(
        experimenter_full_name=opto_model.fiber_data.experimenter_full_name,
        session_type=opto_model.fiber_data.session_type,
        session_start_time=opto_model.fiber_data.session_start_time,
        session_end_time=opto_model.fiber_data.session_end_time,
        rig_id=opto_model.fiber_data.rig_id,
        subject_id=opto_model.fiber_data.subject_id,
        iacuc_protocol=opto_model.fiber_data.iacuc_protocol,
        notes=opto_model.fiber_data.notes,
        data_streams=[data_stream],
        mouse_platform_name="test",
        active_mouse_platform=opto_model.fiber_data.active_mouse_platform,
        anaesthesia=opto_model.fiber_data.anaesthesia,
        animal_weight_post=opto_model.fiber_data.animal_weight_post,
        animal_weight_prior=opto_model.fiber_data.animal_weight_prior,
        stimulus_epochs=[stimulus_epoch]
    )

    with open("./session.json", "w") as f:
        f.write(session.model_dump_json(indent=2))




    





