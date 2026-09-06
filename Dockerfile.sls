# Overlay that makes the cog image dual-mode (cog HTTP on pods, RunPod handler
# on serverless). Build the base with `cog build -t ghcr.io/lee101/pixal3dcog:latest`.
ARG BASE_IMAGE=ghcr.io/lee101/pixal3dcog:latest
FROM ${BASE_IMAGE}
WORKDIR /src
ENTRYPOINT ["/src/entry.sh"]
