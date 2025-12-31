/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#0a0a0a",
                foreground: "#ededed",
                primary: {
                    DEFAULT: "#3b82f6",
                    foreground: "#ffffff",
                },
                secondary: {
                    DEFAULT: "#1f2937",
                    foreground: "#ffffff",
                },
                muted: {
                    DEFAULT: "#171717",
                    foreground: "#a3a3a3",
                },
                accent: {
                    DEFAULT: "#10b981",
                    foreground: "#ffffff",
                },
            },
        },
    },
    plugins: [],
}
