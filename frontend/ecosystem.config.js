module.exports = {
  apps: [
    {
      name: "churn-frontend",
      script: "npm",
      args: "start",
      cwd: "./",
      env: {
        NODE_ENV: "production",
        PORT: 4005,
      },
    },
  ],
};
