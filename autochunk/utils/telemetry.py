
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from .logger import logger

_tracer_initialized = False

def init_telemetry(endpoint: str = "http://localhost:6006/v1/traces", enabled: bool = True):
    """
    Initialize OpenTelemetry tracing.
    By default, it points to a local Arize Phoenix instance.
    """
    global _tracer_initialized
    if _tracer_initialized or not enabled:
        return

    try:
        # Quick check: Is the telemetry server even there?
        import urllib.request
        try:
            # We just check the base URL, not the full /v1/traces path necessarily, 
            # but a 200 or even a 404/405 from the host is better than a connection refused.
            base_url = endpoint.split("/v1/")[0]
            with urllib.request.urlopen(base_url, timeout=1):
                pass
        except Exception:
            # If we can't connect in 1 second, it's likely not running.
            # We don't want to spam the user with ConnectionRefusedErrors in background threads.
            logger.info("Telemetry server not found at localhost:6006. Skipping tracing.")
            return

        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _tracer_initialized = True
        logger.info(f"Telemetry initialized, exporting to {endpoint}")
    except Exception as e:
        logger.debug(f"Failed to initialize telemetry: {e}. Running without tracing.")

def get_tracer(name: str = "autochunk"):
    return trace.get_tracer(name)
