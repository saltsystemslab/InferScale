import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

// index.html reads the same key to apply the theme before the first paint.
const storageKey = 'inferscale-docs-theme'

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(storageKey)
    return value === 'light' || value === 'dark' ? value : null
  } catch {
    return null
  }
}

function storeTheme(theme: Theme) {
  try {
    localStorage.setItem(storageKey, theme)
  } catch {
    // Storage is unavailable; the choice lasts until the page reloads.
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.classList.contains('dark') ? 'dark' : 'light',
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  // Follow the system theme until the reader picks one.
  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      if (!storedTheme()) setTheme(query.matches ? 'dark' : 'light')
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    storeTheme(next)
  }

  return { theme, toggleTheme }
}
