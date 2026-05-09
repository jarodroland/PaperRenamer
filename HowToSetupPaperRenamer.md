---
marp: true
theme: uncover
paginate: true
backgroundColor: #fff
size: 16:9
---

# How to Setup PaperRenamer
Automating PDF renaming with Ollama and Gemma

---

## 1. Install Ollama

Choose one of the following methods:

**Standard Install:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Using Homebrew (macOS):**
```bash
brew install ollama
```

---

## 2. Pull the Gemma Model

We use the 4b Gemma model for processing:

```bash
ollama pull gemma4:e4b
```

---

## 3. Create Project Directory & Clone Repo

Set up your workspace and get the code:

```bash
PROJECT_DIR=~/projects/OllamaPaperRenamer
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR
mkdir -p code logs PaperRenameInbox PaperRenameOutbox
git clone https://github.com/jarodroland/PaperRenamer.git code
```

---

## 4. Setup the Conda Environment

Configure the Python environment:

```bash
cd code
conda env create -f environment.yml
```

---

## Option: macOS launchctl (1/2)

To use the bultin launchd service on macos to monitor your inbox and launch the script for any file downloaded:

1. Copy your conda initialization code to `.zprofile` (for zsh) to ensure it runs in non-interactive shells.
2. Create a symbolic link to the `.plist` file:

```bash
ln -s ~/projects/PaperRenamer/code/com.user.PaperRenamer.plist ~/Library/LaunchAgents
```

---

## Option: macOS launchctl (2/2)

Anytime you edit the plist, you must reload it:

**Unload:**
```bash
launchctl unload ~/Library/LaunchAgents/com.user.PaperRenamer.plist
```

**Load:**
```bash
launchctl load ~/Library/LaunchAgents/com.user.PaperRenamer.plist
```

---

## Option: Automator (1/4) - Creation

1. Open **Automator.app**
2. Select **Folder Action**
3. Add **"Run Shell Script"**
4. Set **Shell** to `/bin/zsh`
5. Set **Pass Input** to `as arguments`

---

## Option: Automator (2/4) - Shell Script

Enter the following command (adjust paths as necessary):

```zsh
/opt/homebrew/Caskroom/miniforge/base/envs/ollama/bin/python \
/Users/jarod/projects/OllamaPaperRenamer/code/PaperRenamer.py "$1"
```

---

## Option: Automator (3/4) - Variables & Actions

1. Add **Set Value of Variable**: Name it `FILETOMOVE`
2. Add **Ask for Finder Items**: Select "Ignore this action's input" in Options
3. Add **Set Value of Variable**: Name it `DESTINATION`
4. Add **Get Value of Variable**: Select `FILETOMOVE` (Ignore input in Options)
5. Add **Move Finder Items**: Drag `DESTINATION` variable to the "To:" menu

---

## Option: Automator (4/4) - Finishing up

1. **Save** the workflow.
2. Open **Folder Actions Setup.app**
3. Ensure **"Enable Folder Actions"** is checked.
4. Add your **Inbox folder** to the list and associate it with the saved workflow.

---

# Setup Complete!
Your papers will now be automatically renamed.
