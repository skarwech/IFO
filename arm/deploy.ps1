param(
  [string]$Host = "pi@raspberrypi.local",
  [string]$Dest = "/home/pi/ifo"
)

Write-Host "Syncing arm/ to $Host:$Dest ..."
ssh $Host "mkdir -p $Dest"
scp -r ./arm $Host:$Dest/
scp -r ./src $Host:$Dest/
scp ./requirements.txt $Host:$Dest/
ssh $Host "python3 -m venv $Dest/.venv; source $Dest/.venv/bin/activate; pip install -r $Dest/requirements.txt"
Write-Host "Done. To run: ssh $Host 'source $Dest/.venv/bin/activate && python -m arm.edge_agent --backend http://<backend>:8000 --opcua opc.tcp://<server>:4840 --hybrid'"
