import { useMemo } from "react";
import katex from "katex";

interface Props {
  math: string;
  className?: string;
}

const COMMON_OPTS = {
  throwOnError: false,
  errorColor: "#fb7185",
  strict: false as const,
};

/** Inline math: \( ... \) */
export function Tex({ math, className }: Props) {
  const html = useMemo(
    () => katex.renderToString(math, { ...COMMON_OPTS, displayMode: false }),
    [math]
  );
  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/** Block math: \[ ... \] */
export function TexBlock({ math, className }: Props) {
  const html = useMemo(
    () => katex.renderToString(math, { ...COMMON_OPTS, displayMode: true }),
    [math]
  );
  return (
    <div
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
