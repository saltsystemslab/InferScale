import type { ReactNode } from 'react'
import logo from '../assets/inferscale-logo.svg'
import { paperUrl, repoUrl } from '../site'
import { GitHubIcon } from './Icons'
import { iconButton } from './styles'
import { ThemeToggle } from './ThemeToggle'

export function Header({ menu }: { menu: ReactNode }) {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[90rem] items-center gap-2 px-4 sm:px-6 lg:px-8">
        {menu}
        {/* Raised so the letters, not the peak above them, line up with the nav. */}
        <a
          href="#overview"
          className="shrink-0 -translate-y-1 rounded-md focus-visible:outline-offset-4"
        >
          {/* The navy wordmark would vanish on the dark canvas, so dark mode paints it white. */}
          <img
            src={logo}
            alt="InferScale"
            width={103}
            height={28}
            className="h-7 w-auto dark:brightness-0 dark:invert"
          />
        </a>
        <div className="ml-auto flex items-center gap-1">
          <a
            href={paperUrl}
            className="rounded-lg px-3 py-2 text-sm font-medium text-ink-muted transition-colors hover:bg-surface hover:text-ink"
          >
            Paper
          </a>
          <a href={repoUrl} className={iconButton} aria-label="InferScale on GitHub">
            <GitHubIcon className="size-[18px]" />
          </a>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
