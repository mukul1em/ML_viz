interface DonutDatum {
  label: string;
  value: number;
  color: string;
}

interface DonutProps {
  data: DonutDatum[];
  size?: number;
  strokeWidth?: number;
  centerLabel?: string;
  centerSub?: string;
  className?: string;
}

export default function Donut({
  data,
  size = 140,
  strokeWidth = 22,
  centerLabel,
  centerSub,
  className = "",
}: DonutProps) {
  const total = data.reduce((a, b) => a + b.value, 0) || 1;
  const radius = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className={className}
      role="img"
    >
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill="transparent"
        stroke="rgba(255,255,255,0.05)"
        strokeWidth={strokeWidth}
      />
      {data.map((d, i) => {
        const len = (d.value / total) * circumference;
        const dash = `${len} ${circumference}`;
        const dashOffset = -offset;
        offset += len;
        return (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={radius}
            fill="transparent"
            stroke={d.color}
            strokeWidth={strokeWidth}
            strokeDasharray={dash}
            strokeDashoffset={dashOffset}
            transform={`rotate(-90 ${cx} ${cy})`}
            strokeLinecap="butt"
          />
        );
      })}
      {centerLabel && (
        <text
          x={cx}
          y={cy + (centerSub ? -4 : 4)}
          textAnchor="middle"
          fill="#e6e8ee"
          fontSize={size * 0.18}
          fontWeight={700}
        >
          {centerLabel}
        </text>
      )}
      {centerSub && (
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          fill="#8b94a8"
          fontSize={size * 0.085}
        >
          {centerSub}
        </text>
      )}
    </svg>
  );
}
