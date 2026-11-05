import { useRef } from 'react'
import { cx } from '../cx'
import type { Heading } from '../hooks/useHeadings'
import { CloseIcon, MenuIcon } from './Icons'
import { Nav } from './Nav'
import { iconButton } from './styles'

type MobileNavProps = { headings: Heading[]; activeId: string | undefined }

/** A menu button that opens the navigation in a modal drawer below the lg breakpoint. */
export function MobileNav({ headings, activeId }: MobileNavProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const close = () => dialog.current?.close()

  return (
    <>
      <button
        type="button"
        onClick={() => dialog.current?.showModal()}
        className={cx(iconButton, '-ml-2 lg:hidden')}
        aria-label="Open navigation"
      >
        <MenuIcon className="size-5" />
      </button>
      {/* A click on the dialog element itself lands on the backdrop; Esc is the keyboard equivalent. */}
      {/* oxlint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions */}
      <dialog
        ref={dialog}
        aria-label="Navigation"
        onClick={(event) => event.target === event.currentTarget && close()}
        className="fixed inset-y-0 left-0 m-0 h-dvh max-h-none w-80 max-w-[calc(100vw-3rem)] border-r border-line bg-canvas p-0 text-ink-body transition-[translate,overlay,display] transition-discrete duration-200 ease-out not-open:-translate-x-full backdrop:bg-gray-950/40 backdrop:backdrop-blur-[2px] open:translate-x-0 motion-reduce:transition-none starting:open:-translate-x-full"
      >
        <div className="flex h-full flex-col">
          <div className="flex h-14 shrink-0 items-center justify-between border-b border-line px-4">
            <span className="text-sm font-semibold text-ink">On this page</span>
            <button
              type="button"
              onClick={close}
              className={cx(iconButton, '-mr-2')}
              aria-label="Close navigation"
            >
              <CloseIcon className="size-5" />
            </button>
          </div>
          <div className="overflow-y-auto px-3 py-4">
            <Nav headings={headings} activeId={activeId} onNavigate={close} />
          </div>
        </div>
      </dialog>
    </>
  )
}
