"""Kafka producer (log replay) and consumer (detection service)."""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Iterable

from .parsing import parse_line
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


def produce(
    lines: Iterable[str],
    bootstrap: str,
    topic: str,
    speed: float = 60.0,
    log_format: str = "app",
) -> int:
    """Send lines to Kafka, preserving their original timing divided by ``speed``.

    ``speed=60`` replays one hour of logs in one minute; ``speed=0`` sends as fast as possible.
    """
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20, "compression.type": "lz4"})
    wall_start = time.monotonic()
    first_ts = None
    sent = 0
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if speed > 0:
            record = parse_line(line, log_format)
            if record is not None:
                if first_ts is None:
                    first_ts = record.timestamp
                due = (record.timestamp - first_ts).total_seconds() / speed
                delay = due - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(delay)
        while True:
            try:
                producer.produce(topic, line.encode("utf-8"))
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.5)
        producer.poll(0)
        sent += 1
        if sent % 10_000 == 0:
            logger.info("Produced %s lines", sent)
    producer.flush(30)
    return sent


def consume(pipeline: Pipeline, bootstrap: str, topic: str, group: str) -> None:
    """Run the detection pipeline on a Kafka topic until SIGINT/SIGTERM."""
    from confluent_kafka import Consumer, KafkaError

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([topic])
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("Consuming from %s topic=%s group=%s", bootstrap, topic, group)
    try:
        while running:
            messages = consumer.consume(num_messages=500, timeout=1.0)
            for message in messages:
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        logger.error("Kafka error: %s", message.error())
                    continue
                pipeline.handle_line(message.value().decode("utf-8", errors="replace"))
    finally:
        pipeline.flush()
        consumer.close()
        logger.info("Consumer stopped")
