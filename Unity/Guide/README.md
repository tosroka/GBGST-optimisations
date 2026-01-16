
## Project Creation
![alt text](setup_0.png)

## Scene
### Add some objects to scene
![alt text](setup_1.png)

### Install Barracuda
```
https://github.com/Unity-Technologies/barracuda-release.git
```
![alt text](setup_2.png)
![alt text](setup_3.png)

### Drop GBGST
Copy content of GBGST archive to Assets/GBGST

### Add Controls to "Main Camera"
1. Open "Main Camera" in Inspector (left click).
2. Add Component "FreeCameraMovement"

![alt text](setup_4.png)

### Run Game & Test
Controls (movement):
- WSAD: forward, backward, left, right
- Q: down
- E: up
- mouse: pitch & yaw

### Create Style Transfer FX Object
1. Create empty GameObject
  - right click in Hierarchy
  - "Create Empty"
2. Add "Custom Pass Volume" component to GameObject
3. Add "CopyPassStylizationEffects" to Custom Passes
4. Configure CustomPassStylizationEffects

![alt text](setup_5.png)

### Add Controls To Style Transfer FX
1. Add Component "Evaluation Controls"

![alt text](setup_6.png)

2. Drag "Custom Pass Volume" from GameObject into "Custom Pass Volume" property in Evaluation Controls Script

![alt text](setup_7.png)

### Run Game & Test
Controls (Style)
- R: take screenshot
- F: increment counter
- C: change mode: normal / stylized

Screenshot is saved to project directory eg. `D:/Unity/SIGKNet/Evaluation`, where `D:/Unity/SIGKNet` is project directory.

Screenshot as `<counter>/<mode>.png`, where:
- `<counter>` is current counter value
- `<mode>` is `false` when stylization is off, otherwise `on`.