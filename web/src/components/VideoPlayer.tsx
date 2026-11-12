import { useEffect, useRef } from "react";

export interface SeekRequest {
  t: number; // seconds into the clip
  n: number; // bump to re-seek to the same time
}

/** <video> that seeks when `seek` changes and reports its playback time (for the chart playhead). */
export default function VideoPlayer({
  src,
  seek,
  onTime,
  autoPlayOnSeek = true,
}: {
  src: string;
  seek?: SeekRequest | null;
  onTime?: (t: number) => void;
  autoPlayOnSeek?: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const v = ref.current;
    if (!v || !seek) return;
    const go = () => {
      v.currentTime = Math.max(0, seek.t);
      if (autoPlayOnSeek) v.play().catch(() => {}); // autoplay can be blocked; seeking still works
    };
    if (v.readyState >= 1) go();
    else {
      v.addEventListener("loadedmetadata", go, { once: true });
      return () => v.removeEventListener("loadedmetadata", go);
    }
  }, [seek, src, autoPlayOnSeek]);

  useEffect(() => {
    const v = ref.current;
    if (!v || !onTime) return;
    let raf = 0;
    const tick = () => {
      onTime(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(tick);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const once = () => onTime(v.currentTime);
    v.addEventListener("play", start);
    v.addEventListener("seeked", once);
    v.addEventListener("pause", once);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", start);
      v.removeEventListener("seeked", once);
      v.removeEventListener("pause", once);
    };
  }, [onTime, src]);

  return (
    <video ref={ref} className="video" src={src} controls muted playsInline preload="metadata">
      Your browser can't play this video.
    </video>
  );
}
