
```sh
# if orphaned procs
ps aux | grep python3
pkill -e -u ubuntu python3

# give permissions to access NVIDIA GPU Performance Counters
if ! sudo grep -q "options nvidia NVreg_RestrictProfilingToAdminUsers=0" /etc/modprobe.d/nvidia.conf 2>/dev/null; then
    echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia.conf
fi
sudo reboot

```
