# Check top 20 files/folders taking space
du -sh /local/home/$USER/* 2>/dev/null | sort -hr | head -20

# Check hidden files/folders taking space
du -sh /local/home/$USER/.* 2>/dev/null | grep -v "^\s*4.0K" | sort -hr

# Check .local space usage
du -sh /local/home/$USER/.local 2>&1 | grep -v "Permission denied" | head -1

du -sh /local/home/$USER/.local/share/* 2>&1 | grep -v "Permission denied" | sort -hr | head -15