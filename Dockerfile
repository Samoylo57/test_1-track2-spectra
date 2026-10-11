# syntax=docker/dockerfile:1
FROM python:3.13-slim@sha256:bb2988715db2cf7ace7b53f38f3cffbef7c7046a656bee66245eb0ed386e2e81
LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.title="Track2 Scattering Spectra + PSMC CPU"
LABEL org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache \
    PYTORCH_KERNEL_CACHE_PATH=/tmp/torch-kernels HOME=/tmp \
    QFBENCH_SS_MAX_SCALES=5 QFBENCH_SS_TARGET_HALF_LIFE=252
WORKDIR /app
COPY requirements-submission.txt /build/requirements.txt
COPY requirements-submission-lock.txt /build/constraints.txt
COPY runtime_wheels/*.whl /build/wheels/
RUN python -m pip install --no-cache-dir torch==2.8.0+cpu --constraint /build/constraints.txt --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir --constraint /build/constraints.txt -r /build/requirements.txt /build/wheels/*.whl \
    && python -m pip check \
    && python -m pip freeze > /app/installed-packages.txt \
    && rm -rf /build
COPY submission_agent.py prediction.py generators.py psmc.py embeddings.py scattering_generator.py scattering_cpu.py joint_scattering.py ./
COPY vendor/ ./vendor/
COPY LICENSE ARTIFACT_PROVENANCE.md ./
USER 65534:65534
ENTRYPOINT ["python", "/app/submission_agent.py"]
