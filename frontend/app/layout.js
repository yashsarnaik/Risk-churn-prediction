import "./globals.css";

export const metadata = {
  title: "Patient Churn Prediction — Dashboard",
  description:
    "ML-powered dashboard to predict and monitor patient churn risk in health programs",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
