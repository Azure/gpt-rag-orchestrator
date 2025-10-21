import os
import logging
import logging.config
import platform

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource, SERVICE_INSTANCE_ID, SERVICE_VERSION, SERVICE_NAMESPACE
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from dependencies import get_config
from connectors.appconfig import AppConfigClient

# Custom filter to exclude trace logs
class ExcludeTraceLogsFilter(logging.Filter):
    def filter(self, record):
        filter_out = 'applicationinsights' not in record.getMessage().lower()
        filter_out = filter_out and 'response status' not in record.getMessage().lower()
        filter_out = filter_out and 'transmission succeeded' not in record.getMessage().lower()
        return filter_out

class DebugModeFilter(logging.Filter):
    """Allow records only when the root logger is in DEBUG.

    Used to gate very verbose Azure HTTP pipeline logs so they only appear
    when explicitly requested via LOG_LEVEL=DEBUG.
    """
    def filter(self, record):
        return logging.getLogger().getEffectiveLevel() == logging.DEBUG


class Telemetry:
    """
    Manages logging and the recording of application telemetry.
    """

    log_level : int = logging.WARNING
    azure_log_level : int = logging.WARNING
    azure_http_log_level : int = logging.CRITICAL
    azure_http_logs_disabled : bool = True
    langchain_log_level : int = logging.NOTSET
    api_name : str = None
    telemetry_connection_string : str = None

    @staticmethod
    def configure_basic(config: AppConfigClient):
        # Determine app log level
        level = Telemetry.translate_log_level(config.get('LOG_LEVEL', 'INFO'))

        # Apply base config with force to avoid duplicate handlers (e.g., under uvicorn --reload)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

        # Azure SDK loggers level (default WARNING unless overridden)
        azure_level = Telemetry.translate_log_level(config.get('AZURE_LOG_LEVEL', 'WARNING'))
        for name in (
            "azure",
            "azure.identity",
            "azure.core",
            "azure.monitor",
        ):
            lg = logging.getLogger(name)
            lg.setLevel(azure_level)
            lg.propagate = False
            lg.filters = []
            lg.handlers.clear()
            lg.addHandler(logging.NullHandler())

        # Azure HTTP pipeline logger: completely silent unless AZURE_HTTP_LOG_LEVEL is provided
            try:
                http_level_override = config.get_value('AZURE_HTTP_LOG_LEVEL', default=None, allow_none=True, type=str)
            except Exception:
                http_level_override = None
        if http_level_override is not None:
            http_level = Telemetry.translate_log_level(http_level_override)
            http_disabled = False
        else:
            http_level = logging.CRITICAL
            http_disabled = True

        http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        http_logger.setLevel(http_level)
        http_logger.disabled = http_disabled
        http_logger.propagate = not http_disabled
        http_logger.handlers = []
        http_logger.filters = []
        if http_disabled:
            http_logger.addHandler(logging.NullHandler())
        #logging.getLogger("httpx").setLevel(config.get_value('HTTPX_LOGLEVEL', 'ERROR').upper())
        #logging.getLogger("httpcore").setLevel(config.get_value('HTTPCORE_LOGLEVEL', 'ERROR').upper())
        #logging.getLogger("openai._base_client").setLevel(config.get_value('OPENAI_BASE_CLIENT_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("urllib3").setLevel(config.get_value('URLLIB3_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("urllib3.connectionpool").setLevel(config.get_value('URLLIB3_CONNECTIONPOOL_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("openai").setLevel(config.get_value('OPENAI_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("autogen_core").setLevel(config.get_value('AUTOGEN_CORE_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("autogen_core.events").setLevel(config.get_value('AUTOGEN_EVENTS_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("uvicorn.error").propagate = True
        #logging.getLogger("uvicorn.access").propagate = True


    @staticmethod
    def configure_monitoring(config: AppConfigClient, telemetry_connection_string: str, api_name : str):

        # Try to get the connection string without throwing if config is disabled/unavailable.
        try:
            Telemetry.telemetry_connection_string = config.get(
                telemetry_connection_string,
                default=os.getenv(telemetry_connection_string)
            )
        except Exception:
            Telemetry.telemetry_connection_string = os.getenv(telemetry_connection_string)

        # If we have no connection string, disable telemetry gracefully with a clear message.
        if not Telemetry.telemetry_connection_string:
            reason = (
                "authentication not available (run 'az login' or configure Managed Identity)"
                if getattr(config, "disabled", False)
                else f"missing '{telemetry_connection_string}'"
            )
            logging.info("Telemetry disabled: %s.", reason)
            return

        Telemetry.api_name = api_name
        resource = Resource.create(
            {
                SERVICE_NAME: f"{Telemetry.api_name}",
                SERVICE_NAMESPACE : api_name,
                SERVICE_VERSION: f"1.0.0",
                SERVICE_INSTANCE_ID: f"{platform.node()}"
            })

        # Quiet noisy DEBUG during setup (optional)
        quiet_names = [
            "azure.monitor.opentelemetry",
            "azure.monitor.opentelemetry._configure",
            "opentelemetry",
            "azure.core.pipeline.policies.http_logging_policy",
        ]
        saved = []
        try:
            for name in quiet_names:
                lg = logging.getLogger(name)
                saved.append((lg, lg.level))
                lg.setLevel(logging.WARNING)
            
            # Allow users to opt-in to sending application logs to App Insights.
            # Tracing remains enabled by default.
            disable_logging_export = str(config.get("AZURE_MONITOR_DISABLE_LOGGING", os.getenv("AZURE_MONITOR_DISABLE_LOGGING", "true"))).lower() == "true"

            # Configure Azure Monitor defaults
            configure_azure_monitor(
                connection_string=Telemetry.telemetry_connection_string,
                disable_offline_storage=True,
                disable_metrics=True,
                disable_tracing=False,
                disable_logging=disable_logging_export,
                resource=resource
            )
        finally:
            # Restore original levels
            for lg, lvl in saved:
                try:
                    lg.setLevel(lvl)
                except Exception:
                    pass

    #Configure telemetry logging (console + optional Azure Monitor logging via SDK)
        Telemetry.configure_logging(config)

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    @staticmethod
    def get_tracer(name: str) -> Tracer:
        return trace.get_tracer(name)

    @staticmethod
    def record_exception(span: Span, ex: Exception):
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ex)

    @staticmethod
    def translate_log_level(log_level: str) -> int:
        """Map a variety of input strings to logging levels.

        Accepts standard names (DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET),
        common synonyms (Trace -> DEBUG, Information -> INFO), and integers.
        Case-insensitive.
        """
        if log_level is None:
            return logging.INFO
        if isinstance(log_level, int):
            return int(log_level)
        s = str(log_level).strip()
        # Try standard logging names first
        std = getattr(logging, s.upper(), None)
        if isinstance(std, int):
            return std
        # Synonyms
        synonyms = {
            "trace": logging.DEBUG,
            "information": logging.INFO,
        }
        return synonyms.get(s.lower(), logging.INFO)

    @staticmethod
    def configure_logging(config: AppConfigClient):
        # Resolve log levels with robust parsing; default INFO for app, WARNING for azure SDK
        Telemetry.log_level = Telemetry.translate_log_level(
            config.get("LOG_LEVEL", default="INFO")
        )
        Telemetry.azure_log_level = Telemetry.translate_log_level(
            config.get("AZURE_LOG_LEVEL", default="WARNING")
        )
        try:
            http_level_override = config.get_value("AZURE_HTTP_LOG_LEVEL", default=None, allow_none=True, type=str)
        except Exception:
            http_level_override = None
        if http_level_override is not None:
            Telemetry.azure_http_log_level = Telemetry.translate_log_level(http_level_override)
            Telemetry.azure_http_logs_disabled = False
        else:
            Telemetry.azure_http_log_level = logging.CRITICAL
            Telemetry.azure_http_logs_disabled = True

        enable_console_logging = str(config.get("ENABLE_CONSOLE_LOGGING", default='true')).lower()

        handlers = []

        if Telemetry.log_level == logging.DEBUG:
            handlers.append(logging.StreamHandler())

        #Logging configuration
        LOGGING = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
                },
                'standard': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
                },
                'azure': {
                    'format': '%(name)s: %(message)s'
                },
                'error': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s %(process)d::%(module)s|%(lineno)s:: %(message)s'
                }
            },
            'handlers': {
                'default': {
                    'level': Telemetry.log_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'filters' : ['exclude_trace_logs'],
                    'stream': 'ext://sys.stdout',
                },
                'console': {
                    'level': Telemetry.log_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'filters' : ['exclude_trace_logs'],
                    'stream': 'ext://sys.stdout'
                },
                "azure": {
                    'formatter': 'azure',
                    'level': Telemetry.log_level,
                    "class": "opentelemetry.sdk._logs.LoggingHandler",
                    'filters' : ['exclude_trace_logs'],
                }
            },
            'filters': {
                'exclude_trace_logs': {
                    '()': 'telemetry.ExcludeTraceLogsFilter',
                },
            },
            'loggers': {
                # Keep Azure SDK logs quiet unless explicitly raised
                'azure': {
                    'level': Telemetry.azure_log_level,
                    'handlers': [],
                    'propagate': False,
                },
                # Gate the very chatty HTTP pipeline logger
                'azure.core.pipeline.policies.http_logging_policy': {
                    'level': Telemetry.azure_http_log_level,
                    'handlers': [] if Telemetry.azure_http_logs_disabled else ['console'],
                    'propagate': not Telemetry.azure_http_logs_disabled,
                },
                '': {
                    'handlers': ['console'],
                    'level': Telemetry.log_level,
                    'filters': ['exclude_trace_logs'],
                },
            },
            "root": {
                "handlers": ["azure", "console"],
                "level": Telemetry.log_level,
            }
        }

        #remove console if prod env (cut down on duplicate log data)
        if enable_console_logging != 'true':
            LOGGING['root']['handlers'] = ["azure"]

        #set the logging configuration
        logging.config.dictConfig(LOGGING)

    @staticmethod
    def log_log_level_diagnostics(config: AppConfigClient) -> None:
        """Log the resolved LOG_LEVEL and the effective root logger level.

        Prefers the environment variable LOG_LEVEL, then App Configuration,
        otherwise defaults to INFO. Keeps main.py minimal.
        """
        try:
            lvl_env = os.getenv("LOG_LEVEL")
            if lvl_env:
                src = "env"
                resolved = lvl_env.strip().upper()
            else:
                try:
                    cfg_val = config.get("LOG_LEVEL", None)
                except Exception:
                    cfg_val = None
                if cfg_val:
                    src = "appconfig"
                    resolved = str(cfg_val).strip().upper()
                else:
                    src = "default"
                    resolved = "INFO"

            logging.getLogger().info("Resolved LOG_LEVEL=%s (source=%s)", resolved, src)
            logging.getLogger().info(
                "Effective root logger level: %s",
                logging.getLevelName(logging.getLogger().getEffectiveLevel()),
            )
        except Exception:
            # Best effort only
            pass