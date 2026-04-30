/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Claude 暖色调
        cream: {
          50: "#FDFCF8",
          100: "#FAF9F5",
          200: "#F4F1EA",
          300: "#E8E2D4",
        },
        terracotta: {
          400: "#E89A82",
          500: "#D97757", // 主色
          600: "#C9623F", // hover
          700: "#A84F30",
        },
        ink: {
          400: "#6B6B6B",
          500: "#4A4A4A",
          600: "#2C2C2C",
          700: "#1A1A1A",
          900: "#1A0617", // 深紫黑（dark mode）
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['"Source Serif Pro"', 'Georgia', 'serif'],
      },
      boxShadow: {
        soft: "0 1px 3px rgba(40, 30, 20, 0.06), 0 1px 2px rgba(40, 30, 20, 0.04)",
        warm: "0 4px 12px rgba(217, 119, 87, 0.12)",
      },
      borderRadius: {
        DEFAULT: "0.5rem",
      },
    },
  },
  plugins: [],
};
