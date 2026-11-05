import { useTheme } from '../theme'
import { MoonIcon, SunIcon } from './Icons'
import { iconButton } from './styles'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={iconButton}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      {theme === 'dark' ? (
        <SunIcon className="size-[18px]" />
      ) : (
        <MoonIcon className="size-[18px]" />
      )}
    </button>
  )
}
