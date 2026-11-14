import { useEffect, useRef, useState } from 'react';

/** Animate a number toward its target (ease-out), for counters beside bars. */
export function useTween(target: number, durationMs = 1100): number {
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  const shownRef = useRef(target);
  shownRef.current = shown;

  useEffect(() => {
    from.current = shownRef.current;
    if (from.current === target) return;
    if (typeof window === 'undefined' || typeof window.requestAnimationFrame !== 'function') {
      setShown(target);
      return;
    }
    const start = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(from.current + (target - from.current) * eased);
      if (t < 1) raf = window.requestAnimationFrame(step);
    };
    raf = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return shown;
}
