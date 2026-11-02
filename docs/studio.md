# Scout Studio

[README](../README.md) · [Development](development.md)

Scout Studio provides a native desktop window for the complete match workflow.
Open **Scout Studio** from the Linux Applications menu after the one-time setup
below. Day-to-day use needs no terminal or Python knowledge.

- Browse and search matches in a local library, or create a new match.
- Select a match and click **Delete match…** below **New match** to remove it
  from the library list. Confirm the dialog; unsaved edits use the usual save/discard
  prompt. Footage, field setup, saved reviews and exports stay on disk. To restore
  the match, delete `media/<match>/.scout-removed` in your library folder (enable
  hidden files in your file manager), then reopen the library. Deletion is disabled
  while the match is loading or processing, or footage is importing. Its name stays
  reserved while its media folder exists.
- Import camera videos with the file picker or drag them into the window.
  Imports copy the originals, report progress, and reject filename collisions.
- Select each video, scrub to a reference frame, and click around its field.
  Drag corners to adjust, right-click to remove, then **Save field**.
- Click **Find plays**. The processing queue runs one match at a time while you
  keep browsing. It also observes an existing W3 batch and waits for it to finish.
  **Queue** (Ctrl+0) opens the single app-wide queue in a separate, nonblocking
  panel; starting a job keeps you on the match you are viewing.
- Watch the **entire match** in **Review & export**, including all footage between
  detected plays. Playback automatically continues into the next camera file.
  Scrub anywhere on the full-match slider or marker timeline. The timeline shows
  a sliding 30-second window by default; selecting a play focuses on its range
  with context so the handles stay easy to grab. **Scroll** slides through time;
  **Ctrl+scroll** zooms around the pointer. Choose 15s/30s/60s/2min/5min windows,
  drag the scrollbar, or use **Fit match** for the overview.
- Review puts the video first, with a compact play inspector at the side.
  **Alt+6** hides the inspector for more viewing space; **Ctrl+B** toggles the match
  library. The video retains its aspect ratio as the workspace resizes.
- **Shift + drag** on the timeline creates and saves a play at the length you
  select, in either direction and across camera-file boundaries. A live preview
  shows its bounds and duration; **Esc** cancels. Default labels use the start
  time, such as `Play 00:12:34.560`, and follow boundary changes. Custom names
  are kept. Drag the middle of a play to move the whole block without changing
  its length, or drag either edge handle to resize it. Undo restores either edit.
- Click the first or second half of a marker to seek to its start or end.
  Adjust start/end times and labels, drag the
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
  palette is available in **Light mode** and **Dark mode**, remembered between
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
  Use **<N>** anywhere in the pattern, such as `Saints_<N>_offense`, choose the
  first number and digit width (`007`), or keep automatic padding. Preferences
  are remembered per match; original play-number/time names are also available.
  MP4s are created from the saved review. The **…** menu holds
  refresh, output-folder and optional Shotcut actions. The review workflow is built in.

One-time setup for a source checkout:

```bash
.venv/bin/python -m pip install -e '.[desktop]'
.venv/bin/python scripts/install_studio.py
```

You can also launch with `./start-studio.sh` or `scout-studio`. Select another
library with **Change library**; the app remembers your choice. A library is a
folder containing `media/`, `fields/`, `out/`, and optional `scout.json` settings.
New libraries use this checkout's included play model. The first detection run
may download YOLO weights. Large imports need space for a copy of the footage.

The desktop dependency is optional; existing command-line usage is unchanged.
The Linux menu launcher uses this checkout's installed environment. This is not
a standalone installer for other computers; those need the one-time setup too.
