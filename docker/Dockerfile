FROM ubuntu:22.04 AS media-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG VMAF_REF=v3.0.0
ARG FFMPEG_REF=n6.1.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git libx264-dev meson nasm ninja-build pkg-config python3 xxd && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${VMAF_REF}" https://github.com/Netflix/vmaf.git /tmp/vmaf && \
    meson setup /tmp/vmaf/libvmaf/build /tmp/vmaf/libvmaf \
      --prefix=/opt/media --libdir=lib --buildtype=release \
      -Denable_tests=false -Denable_docs=false -Dbuilt_in_models=true && \
    meson compile -C /tmp/vmaf/libvmaf/build && \
    meson install -C /tmp/vmaf/libvmaf/build

RUN git clone --depth 1 --branch "${FFMPEG_REF}" https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    PKG_CONFIG_PATH=/opt/media/lib/pkgconfig ./configure \
      --prefix=/opt/media --libdir=/opt/media/lib --shlibdir=/opt/media/lib \
      --enable-shared --disable-static --enable-gpl --enable-libx264 --enable-libvmaf \
      --disable-debug --disable-doc --disable-ffplay \
      --extra-cflags=-I/opt/media/include \
      --extra-ldflags='-L/opt/media/lib -Wl,-rpath,/opt/media/lib' && \
    make -j"$(nproc)" && make install

COPY cpp /tmp/cpp
RUN PKG_CONFIG_PATH=/opt/media/lib/pkgconfig g++ -O3 -std=c++17 /tmp/cpp/roi_encoder.cpp \
      -o /opt/media/bin/roi_encoder \
      $(PKG_CONFIG_PATH=/opt/media/lib/pkgconfig pkg-config --cflags --libs libavformat libavcodec libavutil libswscale) \
      -Wl,-rpath,/opt/media/lib

FROM ubuntu:22.04 AS pipeline-test

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgomp1 libx264-163 p7zip-full python3-numpy python3-pandas \
    python3-pip python3-pytest python3-tqdm python3-yaml && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir gymnasium==0.29.1 opencv-python-headless==4.10.0.84
COPY --from=media-builder /opt/media /opt/media
ENV PATH=/opt/media/bin:${PATH} LD_LIBRARY_PATH=/opt/media/lib
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src src
COPY tests tests
RUN ffmpeg -hide_banner -filters 2>&1 | grep -q libvmaf && \
    PYTHONPATH=/workspace/src python3 -m pytest tests

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG YOLOV5_REF=v7.0
ARG YOLOV5_COMMIT=915bbf294bb74c859f0b41f1c23bc395014ea679
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libgomp1 libx264-163 p7zip-full && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${YOLOV5_REF}" https://github.com/ultralytics/yolov5.git /opt/yolov5 && \
    test "$(git -C /opt/yolov5 rev-parse HEAD)" = "${YOLOV5_COMMIT}" && \
    git config --system --add safe.directory /opt/yolov5

WORKDIR /workspace
COPY pyproject.toml /tmp/vcu-surrogate/pyproject.toml
COPY src /tmp/vcu-surrogate/src
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torchvision==0.17.2 && \
    python -m pip install --no-cache-dir /tmp/vcu-surrogate && \
    rm -rf /tmp/vcu-surrogate

COPY --from=media-builder /opt/media /opt/media
ENV PATH=/opt/media/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/media/lib \
    VCU_PROJECT_ROOT=/workspace \
    PYTHONPATH=/workspace/src:/opt/yolov5 \
    YOLOV5_ROOT=/opt/yolov5 \
    YOLOv5_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["vcu-surrogate"]
