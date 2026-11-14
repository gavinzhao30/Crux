import { useEffect, useState } from 'react';

/** The root font size in px; chart text is sized from it so it scales with the screen. */
export function useRootPx(): number {
  const read = () =>
    typeof window === 'undefined'
      ? 16
      : parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
  const [px, setPx] = useState(read);
  useEffect(() => {
    const onResize = () => setPx(read());
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return px;
}
