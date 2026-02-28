/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        "persian-blue": {
          50: "#f0f5fe",
          100: "#dde7fc",
          200: "#c2d6fb",
          300: "#98bcf8",
          400: "#6899f2",
          500: "#4574ec",
          600: "#2f56e1",
          700: "#2845d6",
          800: "#2538a8",
          900: "#243484",
          950: "#1a2251"
        }
      }
    }
  },
  plugins: []
};
