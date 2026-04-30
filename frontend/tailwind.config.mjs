/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      keyframes: {
        // Fade kèm slide nhẹ — cho step rows / details panel khi mới xuất hiện
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        // Fade-in mềm hơn (không di chuyển) — cho preview list, JSON panel
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      animation: {
        'fade-in-up': 'fade-in-up 220ms ease-out both',
        'fade-in': 'fade-in 180ms ease-out both',
      },
    },
  },
  plugins: [],
}
