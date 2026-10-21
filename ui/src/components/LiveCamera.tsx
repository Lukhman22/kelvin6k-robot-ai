/**
 * Live camera view.
 *
 * Primary path is the MJPEG stream (one long-lived connection, real-time).
 * If the browser doesn't render it within a few seconds -- MJPEG can be blocked
 * by proxies, extensions, or the per-origin connection cap -- it automatically
 * falls back to polling /api/camera/snapshot, which uses short-lived requests
 * and works everywhere.
 *
 * The current mode is shown, and if no frame can be obtained at all the real
 * error is displayed rather than claiming LIVE.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, cameraSnapshotUrl, cameraStreamUrl } from "../api";
import type { CameraInfo, DetectedEntity } from "../types";
import { ProvenanceBadge } from "./Field";

interface Props {
  camera: CameraInfo;
  title?: string;
  subtitle?: string;
  streamUrl?: string | (() => string);
  snapshotUrl?: string | (() => string);
  altText?: string;
  peopleCount?: number | null;
  detections?: DetectedEntity[];
  /** Set true to render client-side SVG bounding boxes. Defaults to false because backend MJPEG already renders annotations directly at 15 FPS without polling lag. */
  showSvgOverlay?: boolean;
}

type Mode = "stream" | "snapshot" | "failed";

const STREAM_TIMEOUT_MS = 4000;
const SNAPSHOT_INTERVAL_MS = 250; // ~4 fps

export function LiveCamera({
  camera,
  title = "Live Camera",
  subtitle,
  streamUrl = cameraStreamUrl,
  snapshotUrl = cameraSnapshotUrl,
  altText = "Live camera view of the printing area",
  peopleCount,
  detections,
  showSvgOverlay = false,
}: Props) {
  const isLive = camera.status === "live" || camera.status === "reconnecting";

  const [mode, setMode] = useState<Mode>("stream");
  const [streamKey, setStreamKey] = useState(0);
  const [snapshotSrc, setSnapshotSrc] = useState<string | null>(null);
  const [failureDetail, setFailureDetail] = useState<string | null>(null);
  const [frameCount, setFrameCount] = useState(0);

  const loadedRef = useRef(false);
  const timeoutRef = useRef<number | null>(null);
  const pollRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [videoLayout, setVideoLayout] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
    srcW: number;
    srcH: number;
  } | null>(null);

  const updateLayout = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;

    const img = imgRef.current;
    const srcW = img?.naturalWidth || (camera.resolution && camera.resolution[0]) || 1280;
    const srcH = img?.naturalHeight || (camera.resolution && camera.resolution[1]) || 720;
    if (srcW <= 0 || srcH <= 0) return;

    const videoAspect = srcW / srcH;
    const containerAspect = cw / ch;

    let renderW = cw;
    let renderH = ch;
    let offsetX = 0;
    let offsetY = 0;

    if (containerAspect > videoAspect) {
      // Container is wider than video (pillarbox: black bars on left/right)
      renderH = ch;
      renderW = ch * videoAspect;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    } else {
      // Container is taller than video (letterbox: black bars on top/bottom)
      renderW = cw;
      renderH = cw / videoAspect;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    }

    setVideoLayout({
      left: Math.round(offsetX),
      top: Math.round(offsetY),
      width: Math.round(renderW),
      height: Math.round(renderH),
      srcW,
      srcH,
    });
  }, [camera.resolution]);

  useEffect(() => {
    updateLayout();
    const container = containerRef.current;
    if (!container) return;

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => {
        updateLayout();
      });
      ro.observe(container);
    }

    window.addEventListener("resize", updateLayout);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", updateLayout);
    };
  }, [updateLayout]);

  const snapshotUrlRef = useRef(snapshotUrl);
  snapshotUrlRef.current = snapshotUrl;

  const resolvedStreamUrl = typeof streamUrl === "function" ? streamUrl() : streamUrl;

  const clearTimers = useCallback(() => {
    if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
    if (pollRef.current) window.clearTimeout(pollRef.current);
    timeoutRef.current = null;
    pollRef.current = null;
    inFlightRef.current = false;
  }, []);

  const startSnapshotPolling = useCallback(() => {
    if (pollRef.current) return;
    const tick = async () => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const getter = snapshotUrlRef.current;
        const url = typeof getter === "function" ? getter() : getter;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`snapshot ${response.status}`);
        const blob = await response.blob();
        setSnapshotSrc((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return URL.createObjectURL(blob);
        });
        setFrameCount((n) => n + 1);
        setFailureDetail(null);
      } catch (error) {
        setFailureDetail(error instanceof Error ? error.message : String(error));
      } finally {
        inFlightRef.current = false;
        if (pollRef.current !== null) {
          pollRef.current = window.setTimeout(() => void tick(), SNAPSHOT_INTERVAL_MS);
        }
      }
    };
    pollRef.current = window.setTimeout(() => void tick(), 0);
  }, []);

  // (Re)start the pipeline whenever the camera becomes available.
  useEffect(() => {
    clearTimers();
    loadedRef.current = false;
    setFailureDetail(null);

    if (!isLive) {
      setMode("failed");
      return;
    }

    setMode("stream");
    setStreamKey((key) => key + 1);

    // If the MJPEG <img> hasn't fired onLoad by now, switch to snapshots.
    timeoutRef.current = window.setTimeout(() => {
      if (!loadedRef.current) {
        setMode("snapshot");
        startSnapshotPolling();
      }
    }, STREAM_TIMEOUT_MS);

    return clearTimers;
  }, [isLive, clearTimers, startSnapshotPolling]);

  useEffect(() => {
    if (mode === "snapshot") startSnapshotPolling();
    if (mode !== "snapshot" && pollRef.current) {
      window.clearTimeout(pollRef.current);
      pollRef.current = null;
    }
  }, [mode, startSnapshotPolling]);

  useEffect(() => () => {
    clearTimers();
    if (snapshotSrc) URL.revokeObjectURL(snapshotSrc);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retry = () => {
    clearTimers();
    loadedRef.current = false;
    setFailureDetail(null);
    setMode("stream");
    setStreamKey((key) => key + 1);
    timeoutRef.current = window.setTimeout(() => {
      if (!loadedRef.current) {
        setMode("snapshot");
        startSnapshotPolling();
      }
    }, STREAM_TIMEOUT_MS);
  };

  const [width, height] = camera.resolution ?? [0, 0];

  const count = peopleCount;
  const hasPeopleCount = typeof count === "number";
  const isDetected = typeof count === "number" && count > 0;
  const peopleCountLabel =
    count === 1
      ? "1 Human Detected"
      : typeof count === "number" && count > 1
      ? `${count} Humans Detected`
      : "No Humans Detected";

  return (
    <section className="panel panel-camera">
      <header className="panel-header">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <h2>{title}</h2>
            {hasPeopleCount ? (
              <span
                className={`people-count-badge ${isDetected ? "people-count-detected" : "people-count-none"}`}
                role="status"
                aria-live="polite"
              >
                <span className={`status-dot ${isDetected ? "status-live" : "status-bad"}`} />
                {peopleCountLabel}
              </span>
            ) : null}
          </div>
          {subtitle ? (
            <div style={{ fontSize: "11px", color: "var(--muted)", marginTop: "2px" }}>
              {subtitle}
            </div>
          ) : null}
        </div>
        <div className="panel-header-meta">
          <span className={`status-dot status-${isLive ? "ok" : "bad"}`} />
          <span className="camera-status-text">{camera.status.toUpperCase()}</span>
          {mode === "snapshot" ? <span className="camera-mode">SNAPSHOT MODE</span> : null}
          <ProvenanceBadge provenance={camera.provenance} />
        </div>
      </header>

      <div
        ref={containerRef}
        className="camera-frame"
        style={{ position: "relative", overflow: "hidden" }}
      >
        {isLive && mode === "stream" ? (
          <img
            ref={imgRef}
            key={streamKey}
            className="camera-stream"
            src={`${resolvedStreamUrl}?k=${streamKey}`}
            alt={altText}
            onLoad={() => {
              loadedRef.current = true;
              setFrameCount((n) => n + 1);
              updateLayout();
            }}
            onError={() => {
              setMode("snapshot");
              startSnapshotPolling();
            }}
          />
        ) : isLive && mode === "snapshot" && snapshotSrc ? (
          <img
            ref={imgRef}
            className="camera-stream"
            src={snapshotSrc}
            alt={altText}
            onLoad={() => {
              updateLayout();
            }}
          />
        ) : (
          <div className="camera-placeholder">
            <p className="camera-placeholder-title">
              {isLive ? "Connecting to video…" : "No live video"}
            </p>
            <p className="camera-placeholder-reason">
              {camera.error ??
                failureDetail ??
                (isLive
                  ? `Requesting frames from ${API_BASE}/api/camera/…`
                  : `Camera status: ${camera.status}.`)}
            </p>
            <button className="btn btn-small" onClick={retry}>Retry</button>
          </div>
        )}

        {showSvgOverlay && detections && detections.length > 0 && isLive ? (() => {
          const defaultSrcW = (camera.resolution && camera.resolution[0]) || 1280;
          const defaultSrcH = (camera.resolution && camera.resolution[1]) || 720;
          const layout = videoLayout || {
            left: 0,
            top: 0,
            width: "100%",
            height: "100%",
            srcW: defaultSrcW,
            srcH: defaultSrcH,
          };
          const clipId = `cam-clip-${title.toLowerCase().replace(/[^a-z0-9]/g, "-")}`;

          return (
            <svg
              className="camera-overlay-svg"
              viewBox={`0 0 ${layout.srcW} ${layout.srcH}`}
              preserveAspectRatio="xMidYMid meet"
              style={{
                position: "absolute",
                left: typeof layout.left === "number" ? `${layout.left}px` : layout.left,
                top: typeof layout.top === "number" ? `${layout.top}px` : layout.top,
                width: typeof layout.width === "number" ? `${layout.width}px` : layout.width,
                height: typeof layout.height === "number" ? `${layout.height}px` : layout.height,
                pointerEvents: "none",
                overflow: "hidden",
              }}
            >
              <defs>
                <clipPath id={clipId}>
                  <rect x="0" y="0" width={layout.srcW} height={layout.srcH} />
                </clipPath>
              </defs>
              <g clipPath={`url(#${clipId})`}>
                {detections.map((d, i) => {
                  const [bx, by, bw, bh] = d.box;
                  const maskPoints = d.mask && d.mask.length > 2
                    ? d.mask.map((pt) => `${pt[0]},${pt[1]}`).join(" ")
                    : null;
                  const isPerson = d.class_name.toLowerCase() === "person";
                  const strokeColor = isPerson ? "#22c55e" : "#38bdf8";
                  const fillColor = isPerson ? "rgba(34, 197, 94, 0.32)" : "rgba(56, 189, 248, 0.28)";
                  const idPrefix = d.track_id != null ? `#${d.track_id} ` : "";
                  const labelText = `${idPrefix}${d.class_name} ${(d.confidence * 100).toFixed(0)}% (${d.width_px}×${d.height_px}px)`;

                  return (
                    <g key={i}>
                      {maskPoints ? (
                        <polygon
                          points={maskPoints}
                          fill={fillColor}
                          stroke={strokeColor}
                          strokeWidth={2}
                          strokeLinejoin="round"
                          strokeLinecap="round"
                        />
                      ) : null}
                      <rect
                        x={bx}
                        y={by}
                        width={bw}
                        height={bh}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={1.5}
                      />
                      <g transform={`translate(${bx}, ${Math.max(18, by - 4)})`}>
                        <rect
                          x={0}
                          y={-15}
                          width={Math.max(110, labelText.length * 7 + 10)}
                          height={18}
                          fill="rgba(15, 23, 42, 0.88)"
                          stroke={strokeColor}
                          strokeWidth={1}
                          rx={3}
                        />
                        <text
                          x={5}
                          y={-2}
                          fill="#ffffff"
                          fontSize={11}
                          fontWeight="600"
                          fontFamily="monospace"
                        >
                          {labelText}
                        </text>
                      </g>
                    </g>
                  );
                })}
              </g>
            </svg>
          );
        })() : null}
      </div>

      {detections && detections.length > 0 ? (
        <div className="camera-detections-list" style={{ padding: "8px 12px", borderTop: "1px solid var(--border, #333)", fontSize: "11px" }}>
          <div style={{ fontWeight: 600, color: "var(--muted, #888)", marginBottom: "4px" }}>
            Detected Entities ({detections.length})
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {detections.map((d, idx) => (
              <span
                key={idx}
                className="detection-chip"
                style={{
                  background: "rgba(255, 255, 255, 0.05)",
                  border: `1px solid ${d.class_name.toLowerCase() === "person" ? "rgba(16, 185, 129, 0.4)" : "rgba(56, 189, 248, 0.4)"}`,
                  borderRadius: "4px",
                  padding: "2px 6px",
                  fontFamily: "monospace",
                }}
              >
                <strong style={{ color: d.class_name.toLowerCase() === "person" ? "#10b981" : "#38bdf8" }}>
                  {d.class_name}
                </strong>{" "}
                <span>{(d.confidence * 100).toFixed(0)}%</span>{" "}
                <span style={{ color: "var(--muted, #aaa)" }}>({d.width_px}×{d.height_px}px)</span>
                {d.track_id != null ? <span style={{ marginLeft: "4px", opacity: 0.7 }}>#{d.track_id}</span> : null}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <dl className="camera-meta">
        <div><dt>Source</dt><dd>{camera.source?.toUpperCase() || "—"}</dd></div>
        <div><dt>Resolution</dt><dd>{width && height ? `${width}×${height}` : "—"}</dd></div>
        <div><dt>Backend FPS</dt><dd>{camera.fps ? camera.fps.toFixed(1) : "—"}</dd></div>
        <div><dt>Mode</dt><dd>{mode === "stream" ? "MJPEG" : mode === "snapshot" ? "Snapshot" : "—"}</dd></div>
        <div><dt>Frames</dt><dd>{frameCount}</dd></div>
        {hasPeopleCount ? <div><dt>People</dt><dd>{count}</dd></div> : null}
      </dl>
    </section>
  );
}
