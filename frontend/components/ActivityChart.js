"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;

  return (
    <div
      style={{
        background: "rgba(17, 24, 39, 0.95)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8,
        padding: "12px 16px",
        fontSize: 12,
        color: "#f1f5f9",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 6 }}>{label}</div>
      {payload.map((entry, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 2,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: entry.color,
              display: "inline-block",
            }}
          />
          <span style={{ color: "#94a3b8" }}>{entry.name}:</span>
          <span style={{ fontWeight: 600 }}>{entry.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function ActivityChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="empty-state">
        <div className="icon">📊</div>
        <div className="title">No Activity Data</div>
        <div className="subtitle">No activity records found for this patient.</div>
      </div>
    );
  }

  // Format dates for display (show day/month only)
  const chartData = data.map((d) => ({
    ...d,
    date: d.date.substring(5), // "MM-DD"
    weight: d.weight_logged ? 1 : 0,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
        barCategoryGap="20%"
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.04)"
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tick={{ fill: "#64748b", fontSize: 10 }}
          axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
          tickLine={false}
          interval={2}
        />
        <YAxis
          tick={{ fill: "#64748b", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, color: "#94a3b8", paddingTop: 8 }}
        />
        <Bar
          dataKey="meal_count"
          name="Meals"
          fill="#14b8a6"
          radius={[3, 3, 0, 0]}
          maxBarSize={16}
        />
        <Bar
          dataKey="weight"
          name="Weight"
          fill="#a78bfa"
          radius={[3, 3, 0, 0]}
          maxBarSize={16}
        />
        <Bar
          dataKey="exercise_count"
          name="Exercise"
          fill="#38bdf8"
          radius={[3, 3, 0, 0]}
          maxBarSize={16}
        />
        <Bar
          dataKey="collection_progress"
          name="Collections"
          fill="#fbbf24"
          radius={[3, 3, 0, 0]}
          maxBarSize={16}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
