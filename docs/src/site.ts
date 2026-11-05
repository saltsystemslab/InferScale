export const repoUrl = 'https://github.com/saltsystemslab/InferScale'
export const paperUrl = 'https://arxiv.org/abs/2607.27090'

/** GitHub URL of a repository path on main; paths ending in "/" are directories. */
export function repoPathUrl(path: string): string {
  const kind = path.endsWith('/') ? 'tree' : 'blob'
  return `${repoUrl}/${kind}/main/${path.replace(/\/$/, '')}`
}
