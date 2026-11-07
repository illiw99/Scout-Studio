# Scout Studio

[README](../README.md) · [Development](development.md)

Scout Studio provides a native desktop window for the complete match workflow.
Open **Scout Studio** from the Linux Applications menu after the one-time setup
below. Day-to-day use needs no terminal or Python knowledge.

The welcome screen opens with **Import game**. An empty or unavailable game
offers **Refresh match**; inactive editors stay hidden until footage is ready.
The compact summaries leave more room for the field canvas. Search shows the
number of matching games and a clear message when nothing matches; clearing it
keeps your current match selected.

Use the sliders icon in the header for **Detection settings**. The profile is
saved automatically and applies to new processing jobs. The default model uses
GPU acceleration when available and CPU otherwise. **Find plays** explains any
prerequisite in its tooltip, such as saving a field for every camera file.

The default library index lives in standard application user data. Launching
from another directory does not select another library. Remembered workspaces
and explicit `--workspace` arguments are preserved. Use **Library → Reveal library
folder** or **Back up library index…** to find or copy the registry. **Import
library index…** merges an old index or JSON backup, rejects conflicting names,
and backs up an existing destination index before changing it. This imports
registry entries only: it does not copy footage or inspect game `Scout/` folders.
Imported entries say **Open to check footage** until selected; your current
editor and unsaved changes stay intact. Offline game paths remain registered.

- Click **Import game** and choose the folder containing the game's videos.
  Its folder name becomes the match name. Videos stay in place; nested camera
  folders are included. Each game can live on a different drive or in a different
  folder. The library remembers imported folders across app restarts.
  Empty folders can be imported too; **No videos in this folder** appears under
  the match name. Add footage to the folder, then right-click **Refresh match**.
- Game data lives inside that game's visible `Scout/` folder: `review.mlt` and
  other review files, `fields/` polygons, `cache/` detections, `clips/` exports,
  `desktop/` processing configurations and logs, and `preferences.ini` export
  naming preferences. Generated videos under `Scout/` never become input footage.
  Older `.scout/` and lowercase `scout/` folders must be renamed with the migration
  tool before opening the game. Only `Scout/` is used for game data; generated
  videos left under any of these names are excluded from footage discovery.
  Use **Open output folder** in the match's right-click menu to find these files.
- Right-click a match for refresh, game/output folder, and **Delete match…**
  actions. Deletion removes its library entry and preserves all files. Import
  its folder again to restore it with its saved work. Busy matches cannot be
  deleted; unsaved edits use the usual save/discard prompt. An unavailable drive
  leaves the game listed; reconnect it at the same location and refresh the match.
  If two folders have the same name, rename one before importing it.
- Existing copied matches can also use the new scheme: choose their existing
  `media/<match>` folder with **Import game**. Studio brings saved output, reviews,
  field sidecars and matching detection caches into `Scout/` while retaining the
  originals. Existing `Scout/` work takes priority. Footage stays in place;
  old clip exports share disk storage where hard links are supported. Importing
  a different folder with the same name still reports a collision.
- The right-click menu also offers **New empty match…** for the older copied
  workflow. For these matches, **Import video files (copy)…** or dropping individual
  videos copies originals into the library and rejects filename collisions.
  Their footage stays in `media/`; generated data uses the same per-game `Scout/`
  layout as linked games, shared with the CLI. To restore
  a deleted copied match, remove `media/<match>/.scout-removed` and restart the app.
- The sun/moon icon at the top right switches theme. The sidebar keeps only
  **Import game**, search, and the match list; it has no library-switch button.
  Folder locations are saved atomically in `.scout-library.json` in the Studio
  library folder (the workspace selected at launch). Keep this index when backing
  up the library; game footage and saved work stay in the individual game folders.
- Select each video, scrub to a reference frame, and click around its field.
  Drag corners to adjust, right-click to remove, then **Save field**.
- Click **Find plays**. The processing queue runs one match at a time while you
  keep browsing. It also observes an existing W3 batch and waits for it to finish.
  **Queue** (Ctrl+0) opens the single app-wide queue in a separate, nonblocking
  panel with a themed header and live status/progress badge. Its dock icon moves
  it into the main window; the close icon only hides it, so processing continues.
  Starting a job keeps you on the match you are viewing.
- Watch the **entire match** in **Review & export**, including all footage between
  detected plays. Playback automatically continues into the next camera file.
  Scrub anywhere on the full-match slider or marker timeline. The timeline shows
  a sliding 30-second window by default; selecting a play focuses on its range
  with context so the handles stay easy to grab. **Scroll** slides through time;
  **Ctrl+scroll** zooms around the pointer. Choose 15s/30s/60s/2min/5min windows,
  drag the scrollbar, or use **Fit match** for the overview.
- Review puts the video first, with plays and clip export settings at the side.
  **Alt+6** hides the sidebar for more viewing space; **Ctrl+B** toggles the match
  library. The video retains its aspect ratio as the workspace resizes.
- **Shift + drag** on the timeline creates and saves a play at the length you
  select, in either direction and across camera-file boundaries. A live preview
  shows its bounds and duration; **Esc** cancels. Default labels use the start
  time, such as `Play 00:12:34.560`, and follow boundary changes. Custom names
  are kept. Drag the middle of a play to move the whole block without changing
  its length, or drag either edge handle to resize it. Undo restores either edit.
- Click the first or second half of a marker to seek to its start or end.
  Double-click a play or choose **Edit play** on the timeline to open its exact
  start/end times and label in a separate editor. You can also drag the
  selected marker's timeline edges (snapped to source frames), add missing plays,
  or delete proposals. Playback previews a dragged boundary, pauses during a
  scrub, and resumes afterward only if it was playing beforehand.
  **Save play changes** writes the authoritative
  `review.mlt`; the original is backed up as `review.before-studio.mlt`.
- To mark a missed play, press **M** at its start, use **I** / **O** to set in/out
  points at the cursor, and press **Ctrl+S** to save. **M** edits an existing play
  when the cursor is inside its range. Right-click a timeline play for **Edit**,
  **Replay**, or **Delete**, or select it and press **Delete**. Undo restores it.
- Playback and timeline shortcuts follow [Shotcut's reference](https://www.shotcut.org/howtos/keyboard-shortcuts/).
  **Space** plays/pauses even when a header button has focus. **J/K/L** reverse,
  pause, and play forwards; repeated J or L increases speed through **10×**.
  Space and pause controls reset speed to **1×**.
  Reverse uses paused frame seeks across source files. **Left/Right** step frames;
  holding **K** and tapping **J/L** also steps frames. **Page Up/Down** seek one
  second, with **Shift** for two, **Ctrl** for five, or **Ctrl+Shift** for ten.
  **< / >** navigate plays; **\** repeats and **Ctrl+M** mutes. The speed menu
  supports **0.1×–10×**; Scout's **Alt+− / Alt+=** slow down / speed up.
- **Ctrl+`** opens a normal, resizable video window (or brings it back).
  **F11**, **Ctrl+Shift+F**, or **F** toggle video fullscreen; **Esc** returns to
  the normal window. Double-click the video for fullscreen. Speed and the current
  match stay synchronized between windows. **Ctrl+0** opens Jobs; **Ctrl+Q** quits.
  Shortcuts keep normal text-field behavior; **Ctrl+S** works while editing.
  Unsaved edits prompt before leaving a marker. Film Hub's navy, gray and lavender
  palette has light and dark modes, selected with the top-right sun/moon icon and remembered between
  launches. No Film Hub server is required.
- Playback, marker and timeline controls use compact SVG symbols with accessible
  names and action-only tooltips. Shortcut hints appear only in **?**; press
  **?**, **/**, or **F1** for the scrollable guide. **=/−** zoom, **0** fits the
  match, and **Ctrl+Shift+P** centers on the cursor. Exact position, window length
  and marker bounds stay visible.
- Use **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z) for marker additions, deletions,
  boundary changes and label edits. Saves and undo/redo preserve your viewing
  position and playback. Undo history lasts until you reload or change matches.
- Camera files follow natural filename order, including edited suffixes such as
  `0077a`. For an older saved review, **… → Order camera files by name** moves
  markers with their original source footage and saves an exact original backup.
  A play split across files that become separated is kept as separate ranges.
- **Export clips** opens a naming preview before adding the job to the queue.
  Set **Pre-play** and **Post-play** beneath the plays list to add a fixed
  number of seconds to every clip. The hatched regions with dashed outlines
  show the extra footage around each solid play marker. Padding follows marker
  edits, stops at the first/last frame of available footage, and can cross camera
  file boundaries. Overlapping clips remain separate exports. These values are
  remembered per match and can also be adjusted in the export dialog; canceling
  restores the timeline preview. The saved play boundaries remain unchanged.
  The same sidebar also holds the filename pattern, first number, number width
  and live filename preview. Changes are remembered automatically per match;
  **Export clips** opens the final confirmation with those settings already filled.
  Use **<N>** anywhere in the pattern, such as `Saints_<N>_offense`, choose the
  first number and digit width (`007`), or keep automatic padding. Preferences
  are remembered per match; original play-number/time names are also available.
  MP4s are created from the saved review. The review **…** menu also holds
  refresh, output-folder and optional Shotcut actions. The review workflow is built in.

One-time setup for a source checkout:

```bash
.venv/bin/python -m pip install -e '.[desktop]'
.venv/bin/python scripts/install_studio.py
```

You can also launch with `./start-studio.sh` or `scout-studio`. The checkout's
script and Linux menu shortcut explicitly open this checkout's workspace.
For other launches, Studio uses a `.scout-library.json` in the current directory
before a remembered workspace. Without either, it uses a local `media/` library
or the default Videos library. An explicit
`scout-studio --workspace /path/to/library` always selects that workspace.
Existing copied matches and optional `scout.json` settings still work.
New libraries use the included play model. The first detection run may download
YOLO weights. Folder imports need space only for generated game data; the optional
copied-video workflow also needs space for copies of the footage.

The desktop dependency is optional; existing command-line usage is unchanged.
The Linux menu launcher uses this checkout's installed environment. This is not
a standalone installer for other computers; those need the one-time setup too.
