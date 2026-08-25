# Transcript 03: The project website

**Deliverable:** `https://care-capacity-site.vercel.app` (Vercel project
`alanna2/care-capacity-site`)
**Source:** a separate `website/` folder, React 18 + Vite 5
**Paper reference:** Agentic Analysis, "Website"

This transcript covers the site for both projects, since one site presents
QSS 20 and QSS 45 together. 

Edited excerpt from the session log, abridged as described in the README.

---

## What was asked for

> **Alanna:** Create a personal website concept driven by a "Organic Editorial"
> vibe [...] but professional and for the pharma-type theme, maybe the
> similar blue and orange colors. Leave gaps for figures in all of the tabs
> that make it easy to fill in by hand. No grey text, only black or other colors.

The AI produced the scaffold and a first full draft: `App.jsx` holding
the page content, `Blocks.jsx` holding the reusable components (`Section`,
`Figure`, `Gallery`, `Table`, `Stats`, `Callout`), `styles.css`, and the Vite
configuration.

---

## Where it went wrong

**A deploy failure the local build could not reproduce.**

```
sh: line 1: vite: command not found
Error: Command "vite build" exited with 127
```

The site built cleanly on my machine and failed on Vercel every time. The cause
was that Vercel installs with `NODE_ENV=production`, which makes `npm ci` skip
`devDependencies` and `vite` had been scaffolded into `devDependencies`, which
is where it normally belongs. Reproduced exactly by running
`NODE_ENV=production npm ci`, which installed 5 packages instead of the full
tree. The fix was to move `vite` and `@vitejs/plugin-react` into `dependencies`,
with a note in `package.json` recording why they sit somewhere unusual.

**A stale build directory that hid a redesign** After a restyle, the
site looked unchanged. `npm run build` empties `dist/` first, could not delete
`dist/.DS_Store`, and died there, thus leaving the previous build in place with no
error I noticed. Deleting `dist/` fixed it.

---

## Reflection

What I asked for: a site presenting both projects.

What I accepted: the component structure, the Vite build, and a design pass.

What I rejected: some complete visual designs, the grey-caption convention, and a
large fraction of the first drafted copy.
