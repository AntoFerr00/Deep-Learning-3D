# 3D Deep Learning Network Visualization

This project is a **3D visualization tool** for deep learning network architectures using **OpenGL** and **WinAPI**. The application renders predefined architectures (such as **AlexNet, VGG16, and ResNet18**) or allows users to define their own custom neural networks.

## Features
- Render convolutional layers as **3D boxes**.
- Render fully connected layers as **spheres with interconnections**.
- Interactive camera controls with **mouse rotation**.
- **Predefined networks**: AlexNet, VGG16, ResNet18.
- **Custom network setup** via console input.

## Dependencies
- **GCC (MinGW recommended for Windows)**
- **OpenGL** (`-lopengl32`)
- **GLU** (`-lglu32`)
- **Math library (`-lm`)**
- **Windows API (`-mwindows`)**

## Installation & Execution
### 1. Install MinGW (if not installed)
Ensure that `gcc` is available in your system's PATH. You can install MinGW from:
- https://www.mingw-w64.org/

### 2. Compile the Program
Run the following command in the terminal:
```sh
gcc main.c -o deep3d -mwindows -lopengl32 -lglu32 -lm
```
This will produce an executable named **deep3d.exe**.

### 3. Run the Application
Execute the generated binary:
```sh
./deep3d.exe
```

## Controls
- **Mouse Drag**: Rotate the view.
- **Left Click**: Hold to enable rotation.

## Code Structure
### 1. Core Functions
- `SetupConsole()`: Initializes the console for debugging.
- `SetupNetwork()`: Loads predefined or custom network architecture.
- `InitOpenGL()`: Configures OpenGL for rendering.
- `RenderScene()`: Draws the neural network.

### 2. Network Representation
- **Boxes** represent convolutional layers.
- **Spheres** represent fully connected neurons.
- **Arrows** indicate layer connections.

### 3. Predefined Architectures
- `SetupAlexNet()`
- `SetupVGG16()`
- `SetupResNet18()`

## Custom Network Setup
1. Select **Custom Network** in the menu.
2. Enter the number of layers.
3. Define layer type (`Box` for Conv layers, `FC` for fully connected layers).
4. Specify layer dimensions and color.

## Example Output
A rendered **AlexNet** architecture would look like:
```
[Input] → [Conv1] → [Conv2] → [Conv3] → [Conv4] → [Conv5] → [FC6] → [FC7] → [FC8]
```

## Troubleshooting
- If the window does not appear, ensure **OpenGL drivers** are installed.
- If compilation fails, check that **MinGW** is correctly installed and in the system PATH.

## License
This project is open-source and available for modification and distribution.

---
Developed for 3D visualization of neural networks using OpenGL and WinAPI.