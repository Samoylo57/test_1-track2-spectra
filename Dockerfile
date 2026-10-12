# syntax=docker/dockerfile:1
FROM ghcr.io/samoylo57/test_1-track2-spectra@sha256:cc016aeffd046cd164be90fa0c11fa2bd5fefbf801359f9f4009b43e04246402
COPY submission_agent.py center_features.py center_runtime.py /app/
COPY models/center.json /app/models/center.json
COPY ARTIFACT_PROVENANCE.md LICENSE /app/
ENV QFBENCH_SS_MAX_SCALES=5 QFBENCH_SS_TARGET_HALF_LIFE=none QFBENCH_USE_PSMC=0
LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.title="Track2 Spectra CPU + calibrated centre, width x1.5"
USER 65534:65534
ENTRYPOINT ["python", "/app/submission_agent.py"]
