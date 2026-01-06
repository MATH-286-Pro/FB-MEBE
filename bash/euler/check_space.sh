# check .apptainer space usage
du -sh ~/.apptainer/*

# check user account
sacctmgr show associations user=$USER
