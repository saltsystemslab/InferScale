/// <reference types="vitest/config" />
import mdx from '@mdx-js/rollup'
import rehypeShiki from '@shikijs/rehype'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import rehypeSlug from 'rehype-slug'
import remarkGfm from 'remark-gfm'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Relative asset URLs let the same build serve from any GitHub Pages path.
  base: './',
  plugins: [
    {
      enforce: 'pre',
      ...mdx({
        remarkPlugins: [remarkGfm],
        rehypePlugins: [
          rehypeSlug,
          [
            rehypeShiki,
            {
              themes: { light: 'github-light', dark: 'github-dark' },
              defaultColor: false,
            },
          ],
        ],
      }),
    },
    react({ include: /\.(mdx|ts|tsx)$/ }),
    tailwindcss(),
  ],
  build: {
    rolldownOptions: {
      // Tailwind's scan dominates this small build; the timing warning is not actionable.
      checks: { bundlerTimings: false },
    },
  },
  server: {
    // The paper figures are imported from the repository's figures/ directory.
    fs: { allow: ['..'] },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
