"use client";

import { useMemo } from "react";

export default function RiskGauge({ probability, riskLabel }) {
  const percentage = Math.round((probability || 0) * 100);

  const gaugeColor = useMemo(() => {
    if (percentage >= 60) return "var(--risk-high)";
    if (percentage >= 30) return "var(--risk-medium)";
    return "var(--risk-low)";
  }, [percentage]);

  // SVG arc calculations for a semicircle gauge
  const cx = 110;
  const cy = 120;
  const r = 90;
  const startAngle = Math.PI;        // 180° (left)
  const endAngle = 0;                // 0° (right)
  const totalArc = Math.PI;          // 180° sweep

  // Arc path helper
  const describeArc = (startA, endA) => {
    const x1 = cx + r * Math.cos(startA);
    const y1 = cy - r * Math.sin(startA);
    const x2 = cx + r * Math.cos(endA);
    const y2 = cy - r * Math.sin(endA);
    const largeArc = endA - startA > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  // Full track arc (180°)
  const trackPath = describeArc(startAngle, endAngle);

  // Filled arc based on percentage
  const fillAngle = startAngle - (percentage / 100) * totalArc;
  const fillPath =
    percentage > 0 ? describeArc(startAngle, fillAngle) : "";

  // Circumference for dash animation
  const arcLength = Math.PI * r;
  const filledLength = (percentage / 100) * arcLength;

  return (
    <div className="gauge-container">
      <svg className="gauge-svg" viewBox="0 0 220 135">
        {/* Background track */}
        <path
          d={trackPath}
          className="gauge-track"
        />
        {/* Gradient definition */}
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style={{ stopColor: "var(--risk-low)" }} />
            <stop offset="50%" style={{ stopColor: "var(--risk-medium)" }} />
            <stop offset="100%" style={{ stopColor: "var(--risk-high)" }} />
          </linearGradient>
        </defs>
        {/* Filled arc */}
        {percentage > 0 && (
          <path
            d={trackPath}
            className="gauge-fill"
            style={{
              stroke: gaugeColor,
              strokeDasharray: `${arcLength}`,
              strokeDashoffset: `${arcLength - filledLength}`,
            }}
          />
        )}
      </svg>
      <div className="gauge-value">
        <span className="number" style={{ color: gaugeColor }}>
          {percentage}
        </span>
        <span className="percent">%</span>
      </div>
    </div>
  );
}
