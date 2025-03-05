#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-------------------------
// Constants & Prototypes
//-------------------------

#define PI 3.14159265358979323846
#define MAX_LAYERS 20
#define TEXT_OFFSET_X 3.0f
#define TOP_Y 8.0f
#define LAYER_SPACING 2.0f

// Function prototypes
void SetupConsole(void);
void SetupNetwork(void);
void SetupAlexNet(void);
void SetupVGG16(void);
void SetupResNet18(void);
void SetupCustomNetwork(void);

void SetupPixelFormatForDC(HDC hDC);
void InitOpenGL(void);

void DrawBox(float cx, float cy, float cz, float width, float height, float depth);
void DrawSphere(float x, float y, float z, float radius);
void DrawArrow(float x1, float y1, float z1, float x2, float y2, float z2);
void DrawFullyConnectedLayer(float y, int neuronCount);
void RenderText(const char* text, float x, float y, float z);

void DrawNetwork(void);
void RenderScene(void);

// Window procedure
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

//-------------------------
// Data Structures
//-------------------------

typedef enum {
    LAYER_BOX,   // Representing input/conv layers as boxes
    LAYER_FC     // Representing fully-connected layers as rows of spheres
} LayerType;

typedef struct {
    LayerType type;
    union {
        struct {
            float width;
            float height;
            float depth;
        } box;
        struct {
            int neuronCount;
        } fc;
    };
    float color[3];  // RGB color
    char label[64];  // Text label for the layer
} Layer;

Layer networkLayers[MAX_LAYERS];
int numLayers = 0;

//-------------------------
// Global Variables for OpenGL & Window
//-------------------------

HGLRC hRC = NULL;
HDC   hDC = NULL;
HWND  hWnd = NULL;
HINSTANCE hInstance;

GLUquadric* quadric = NULL; // For drawing spheres and cones
GLuint baseList = 0;         // Display list base for font bitmaps

// Global mouse control variables
float rotX = 0.0f, rotY = 0.0f;
int mouseDown = 0;
int lastMouseX = 0, lastMouseY = 0;

//-------------------------
// Console Setup
//-------------------------

void SetupConsole(void) {
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    printf("Console initialized.\n");
}

//-------------------------
// Predefined Network Setup Functions
//-------------------------

// AlexNet (simplified schematic)
void SetupAlexNet(void) {
    numLayers = 9;
    // Layer 0: Input
    networkLayers[0].type = LAYER_BOX;
    networkLayers[0].box.width = 2.0f;
    networkLayers[0].box.height = 1.0f;
    networkLayers[0].box.depth = 2.0f;
    networkLayers[0].color[0] = 1.0f; networkLayers[0].color[1] = 1.0f; networkLayers[0].color[2] = 1.0f;
    strcpy(networkLayers[0].label, "Input");
    
    // Layer 1: Conv1
    networkLayers[1].type = LAYER_BOX;
    networkLayers[1].box.width = 2.0f;
    networkLayers[1].box.height = 1.0f;
    networkLayers[1].box.depth = 2.0f;
    networkLayers[1].color[0] = 1.0f; networkLayers[1].color[1] = 0.0f; networkLayers[1].color[2] = 0.0f;
    strcpy(networkLayers[1].label, "Conv1");
    
    // Layer 2: Conv2
    networkLayers[2].type = LAYER_BOX;
    networkLayers[2].box.width = 1.8f;
    networkLayers[2].box.height = 1.0f;
    networkLayers[2].box.depth = 1.8f;
    networkLayers[2].color[0] = 0.0f; networkLayers[2].color[1] = 1.0f; networkLayers[2].color[2] = 0.0f;
    strcpy(networkLayers[2].label, "Conv2");
    
    // Layer 3: Conv3
    networkLayers[3].type = LAYER_BOX;
    networkLayers[3].box.width = 1.6f;
    networkLayers[3].box.height = 1.0f;
    networkLayers[3].box.depth = 1.6f;
    networkLayers[3].color[0] = 0.0f; networkLayers[3].color[1] = 0.0f; networkLayers[3].color[2] = 1.0f;
    strcpy(networkLayers[3].label, "Conv3");
    
    // Layer 4: Conv4
    networkLayers[4].type = LAYER_BOX;
    networkLayers[4].box.width = 1.4f;
    networkLayers[4].box.height = 1.0f;
    networkLayers[4].box.depth = 1.4f;
    networkLayers[4].color[0] = 1.0f; networkLayers[4].color[1] = 0.0f; networkLayers[4].color[2] = 1.0f;
    strcpy(networkLayers[4].label, "Conv4");
    
    // Layer 5: Conv5
    networkLayers[5].type = LAYER_BOX;
    networkLayers[5].box.width = 1.2f;
    networkLayers[5].box.height = 1.0f;
    networkLayers[5].box.depth = 1.2f;
    networkLayers[5].color[0] = 0.0f; networkLayers[5].color[1] = 1.0f; networkLayers[5].color[2] = 1.0f;
    strcpy(networkLayers[5].label, "Conv5");
    
    // Layer 6: FC6
    networkLayers[6].type = LAYER_FC;
    networkLayers[6].fc.neuronCount = 5;
    networkLayers[6].color[0] = 1.0f; networkLayers[6].color[1] = 1.0f; networkLayers[6].color[2] = 0.0f;
    strcpy(networkLayers[6].label, "FC6");
    
    // Layer 7: FC7
    networkLayers[7].type = LAYER_FC;
    networkLayers[7].fc.neuronCount = 5;
    networkLayers[7].color[0] = 1.0f; networkLayers[7].color[1] = 0.5f; networkLayers[7].color[2] = 0.0f;
    strcpy(networkLayers[7].label, "FC7");
    
    // Layer 8: FC8
    networkLayers[8].type = LAYER_FC;
    networkLayers[8].fc.neuronCount = 5;
    networkLayers[8].color[0] = 0.5f; networkLayers[8].color[1] = 0.5f; networkLayers[8].color[2] = 0.5f;
    strcpy(networkLayers[8].label, "FC8");
}

// VGG16 (simplified schematic)
void SetupVGG16(void) {
    numLayers = 9;
    strcpy(networkLayers[0].label, "Input");
    networkLayers[0].type = LAYER_BOX;
    networkLayers[0].box.width = 3.0f;
    networkLayers[0].box.height = 2.0f;
    networkLayers[0].box.depth = 3.0f;
    networkLayers[0].color[0] = 1.0f; networkLayers[0].color[1] = 1.0f; networkLayers[0].color[2] = 1.0f;
    
    strcpy(networkLayers[1].label, "ConvBlock1");
    networkLayers[1].type = LAYER_BOX;
    networkLayers[1].box.width = 3.0f;
    networkLayers[1].box.height = 1.5f;
    networkLayers[1].box.depth = 3.0f;
    networkLayers[1].color[0] = 1.0f; networkLayers[1].color[1] = 0.0f; networkLayers[1].color[2] = 0.0f;
    
    strcpy(networkLayers[2].label, "ConvBlock2");
    networkLayers[2].type = LAYER_BOX;
    networkLayers[2].box.width = 2.8f;
    networkLayers[2].box.height = 1.5f;
    networkLayers[2].box.depth = 2.8f;
    networkLayers[2].color[0] = 0.0f; networkLayers[2].color[1] = 1.0f; networkLayers[2].color[2] = 0.0f;
    
    strcpy(networkLayers[3].label, "ConvBlock3");
    networkLayers[3].type = LAYER_BOX;
    networkLayers[3].box.width = 2.6f;
    networkLayers[3].box.height = 1.5f;
    networkLayers[3].box.depth = 2.6f;
    networkLayers[3].color[0] = 0.0f; networkLayers[3].color[1] = 0.0f; networkLayers[3].color[2] = 1.0f;
    
    strcpy(networkLayers[4].label, "ConvBlock4");
    networkLayers[4].type = LAYER_BOX;
    networkLayers[4].box.width = 2.4f;
    networkLayers[4].box.height = 1.5f;
    networkLayers[4].box.depth = 2.4f;
    networkLayers[4].color[0] = 1.0f; networkLayers[4].color[1] = 0.0f; networkLayers[4].color[2] = 1.0f;
    
    strcpy(networkLayers[5].label, "ConvBlock5");
    networkLayers[5].type = LAYER_BOX;
    networkLayers[5].box.width = 2.2f;
    networkLayers[5].box.height = 1.5f;
    networkLayers[5].box.depth = 2.2f;
    networkLayers[5].color[0] = 0.0f; networkLayers[5].color[1] = 1.0f; networkLayers[5].color[2] = 1.0f;
    
    strcpy(networkLayers[6].label, "FC1");
    networkLayers[6].type = LAYER_FC;
    networkLayers[6].fc.neuronCount = 5;
    networkLayers[6].color[0] = 1.0f; networkLayers[6].color[1] = 1.0f; networkLayers[6].color[2] = 0.0f;
    
    strcpy(networkLayers[7].label, "FC2");
    networkLayers[7].type = LAYER_FC;
    networkLayers[7].fc.neuronCount = 5;
    networkLayers[7].color[0] = 1.0f; networkLayers[7].color[1] = 0.5f; networkLayers[7].color[2] = 0.0f;
    
    strcpy(networkLayers[8].label, "FC3");
    networkLayers[8].type = LAYER_FC;
    networkLayers[8].fc.neuronCount = 5;
    networkLayers[8].color[0] = 0.5f; networkLayers[8].color[1] = 0.5f; networkLayers[8].color[2] = 0.5f;
}

// ResNet18 (simplified schematic)
void SetupResNet18(void) {
    numLayers = 7;
    strcpy(networkLayers[0].label, "Input");
    networkLayers[0].type = LAYER_BOX;
    networkLayers[0].box.width = 3.0f;
    networkLayers[0].box.height = 2.0f;
    networkLayers[0].box.depth = 3.0f;
    networkLayers[0].color[0] = 1.0f; networkLayers[0].color[1] = 1.0f; networkLayers[0].color[2] = 1.0f;
    
    strcpy(networkLayers[1].label, "InitialConv");
    networkLayers[1].type = LAYER_BOX;
    networkLayers[1].box.width = 3.0f;
    networkLayers[1].box.height = 1.5f;
    networkLayers[1].box.depth = 3.0f;
    networkLayers[1].color[0] = 1.0f; networkLayers[1].color[1] = 0.0f; networkLayers[1].color[2] = 0.0f;
    
    strcpy(networkLayers[2].label, "ResBlock1");
    networkLayers[2].type = LAYER_BOX;
    networkLayers[2].box.width = 2.8f;
    networkLayers[2].box.height = 1.5f;
    networkLayers[2].box.depth = 2.8f;
    networkLayers[2].color[0] = 0.0f; networkLayers[2].color[1] = 1.0f; networkLayers[2].color[2] = 0.0f;
    
    strcpy(networkLayers[3].label, "ResBlock2");
    networkLayers[3].type = LAYER_BOX;
    networkLayers[3].box.width = 2.6f;
    networkLayers[3].box.height = 1.5f;
    networkLayers[3].box.depth = 2.6f;
    networkLayers[3].color[0] = 0.0f; networkLayers[3].color[1] = 0.0f; networkLayers[3].color[2] = 1.0f;
    
    strcpy(networkLayers[4].label, "ResBlock3");
    networkLayers[4].type = LAYER_BOX;
    networkLayers[4].box.width = 2.4f;
    networkLayers[4].box.height = 1.5f;
    networkLayers[4].box.depth = 2.4f;
    networkLayers[4].color[0] = 1.0f; networkLayers[4].color[1] = 0.0f; networkLayers[4].color[2] = 1.0f;
    
    strcpy(networkLayers[5].label, "ResBlock4");
    networkLayers[5].type = LAYER_BOX;
    networkLayers[5].box.width = 2.2f;
    networkLayers[5].box.height = 1.5f;
    networkLayers[5].box.depth = 2.2f;
    networkLayers[5].color[0] = 0.0f; networkLayers[5].color[1] = 1.0f; networkLayers[5].color[2] = 1.0f;
    
    strcpy(networkLayers[6].label, "FinalFC");
    networkLayers[6].type = LAYER_FC;
    networkLayers[6].fc.neuronCount = 5;
    networkLayers[6].color[0] = 1.0f; networkLayers[6].color[1] = 1.0f; networkLayers[6].color[2] = 0.0f;
}

// Custom network: label each layer as "Layer i"
void SetupCustomNetwork(void) {
    printf("Enter the number of layers (max %d): ", MAX_LAYERS);
    scanf("%d", &numLayers);
    if(numLayers > MAX_LAYERS) numLayers = MAX_LAYERS;
    
    printf("Note: You may separate numbers with spaces or commas.\n");
    
    for (int i = 0; i < numLayers; i++) {
        int typeInput;
        printf("Layer %d: Enter type (0 for box, 1 for fully-connected): ", i);
        scanf("%d", &typeInput);
        if (typeInput == 0) {
            networkLayers[i].type = LAYER_BOX;
            printf("Enter width, height, depth for box: ");
            scanf(" %f%*[ ,]%f%*[ ,]%f", 
                  &networkLayers[i].box.width, 
                  &networkLayers[i].box.height, 
                  &networkLayers[i].box.depth);
        } else {
            networkLayers[i].type = LAYER_FC;
            printf("Enter number of neurons for fully-connected layer: ");
            scanf("%d", &networkLayers[i].fc.neuronCount);
        }
        printf("Enter color (r, g, b, each between 0 and 1): ");
        scanf(" %f%*[ ,]%f%*[ ,]%f", 
              &networkLayers[i].color[0], 
              &networkLayers[i].color[1], 
              &networkLayers[i].color[2]);
        sprintf(networkLayers[i].label, "Layer %d", i);
    }
}

//-------------------------
// Master Network Setup Menu
//-------------------------

void SetupNetwork(void) {
    int choice;
    printf("Choose network option:\n");
    printf("1. Predefined: AlexNet\n");
    printf("2. Predefined: VGG16\n");
    printf("3. Predefined: ResNet18\n");
    printf("4. Custom\n");
    printf("Enter choice (1-4): ");
    scanf("%d", &choice);
    if (choice == 1)
        SetupAlexNet();
    else if (choice == 2)
        SetupVGG16();
    else if (choice == 3)
        SetupResNet18();
    else
        SetupCustomNetwork();
}

//-------------------------
// OpenGL Setup & Utility Functions
//-------------------------

void SetupPixelFormatForDC(HDC hDC) {
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize      = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    
    int pixelFormat = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, pixelFormat, &pfd);
}

void InitOpenGL(void) {
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 800.0/600.0, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
    quadric = gluNewQuadric();
    
    // Create display lists for font bitmaps.
    baseList = glGenLists(96);
    HFONT hFont = CreateFont(-16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                             ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, 
                             ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH, "Arial");
    if(!hFont) {
        printf("Error: Failed to create font.\n");
        exit(EXIT_FAILURE);
    }
    SelectObject(hDC, hFont);
    if(!wglUseFontBitmaps(hDC, 32, 96, baseList)) {
        printf("Error: wglUseFontBitmaps failed.\n");
        exit(EXIT_FAILURE);
    }
}

//-------------------------
// Drawing Primitives
//-------------------------

void DrawBox(float cx, float cy, float cz, float width, float height, float depth) {
    float hw = width / 2.0f;
    float hh = height / 2.0f;
    float hd = depth / 2.0f;
    
    glBegin(GL_QUADS);
    // Front face
    glVertex3f(cx - hw, cy - hh, cz + hd);
    glVertex3f(cx + hw, cy - hh, cz + hd);
    glVertex3f(cx + hw, cy + hh, cz + hd);
    glVertex3f(cx - hw, cy + hh, cz + hd);
    // Back face
    glVertex3f(cx - hw, cy - hh, cz - hd);
    glVertex3f(cx + hw, cy - hh, cz - hd);
    glVertex3f(cx + hw, cy + hh, cz - hd);
    glVertex3f(cx - hw, cy + hh, cz - hd);
    // Left face
    glVertex3f(cx - hw, cy - hh, cz - hd);
    glVertex3f(cx - hw, cy + hh, cz - hd);
    glVertex3f(cx - hw, cy + hh, cz + hd);
    glVertex3f(cx - hw, cy - hh, cz + hd);
    // Right face
    glVertex3f(cx + hw, cy - hh, cz - hd);
    glVertex3f(cx + hw, cy + hh, cz - hd);
    glVertex3f(cx + hw, cy + hh, cz + hd);
    glVertex3f(cx + hw, cy - hh, cz + hd);
    // Top face
    glVertex3f(cx - hw, cy + hh, cz - hd);
    glVertex3f(cx + hw, cy + hh, cz - hd);
    glVertex3f(cx + hw, cy + hh, cz + hd);
    glVertex3f(cx - hw, cy + hh, cz + hd);
    // Bottom face
    glVertex3f(cx - hw, cy - hh, cz - hd);
    glVertex3f(cx + hw, cy - hh, cz - hd);
    glVertex3f(cx + hw, cy - hh, cz + hd);
    glVertex3f(cx - hw, cy - hh, cz + hd);
    glEnd();
}

void DrawSphere(float x, float y, float z, float radius) {
    glPushMatrix();
    glTranslatef(x, y, z);
    gluSphere(quadric, radius, 16, 16);
    glPopMatrix();
}

void DrawArrow(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
    float length = sqrt(dx*dx + dy*dy + dz*dz);
    if (length < 0.0001f)
        return;
    
    float arrowHeadLength = (length > 0.5f) ? 0.5f : 0.5f * length;
    float arrowHeadRadius = 0.2f;
    float shaftLength = length - arrowHeadLength;
    float shaftEndX = x1 + (dx / length) * shaftLength;
    float shaftEndY = y1 + (dy / length) * shaftLength;
    float shaftEndZ = z1 + (dz / length) * shaftLength;
    
    glBegin(GL_LINES);
    glVertex3f(x1, y1, z1);
    glVertex3f(shaftEndX, shaftEndY, shaftEndZ);
    glEnd();
    
    glPushMatrix();
    glTranslatef(shaftEndX, shaftEndY, shaftEndZ);
    float dirX = dx / length, dirY = dy / length, dirZ = dz / length;
    float angle = acos(dirZ) * (180.0f / PI);
    float axisX = -dirY, axisY = dirX;
    if (sqrt(axisX*axisX + axisY*axisY) > 0.0001f)
        glRotatef(angle, axisX, axisY, 0.0f);
    else if (dirZ < 0)
        glRotatef(180, 1, 0, 0);
    gluCylinder(quadric, arrowHeadRadius, 0.0, arrowHeadLength, 12, 3);
    glPopMatrix();
}


void DrawFullyConnectedLayer(float y, int neuronCount) {
    float spacing = 1.0f;
    float startX = -((neuronCount - 1) * spacing) / 2.0f;
    float zOffset = 0.5f;
    for (int i = 0; i < neuronCount; i++) {
        float x = startX + i * spacing;
        float z = (i % 2 == 0) ? -zOffset : zOffset;
        DrawSphere(x, y, z, 0.3f);
    }
    for (int i = 0; i < neuronCount; i++) {
        for (int j = 0; j < neuronCount; j++) {
            float x1 = startX + i * spacing;
            float z1 = (i % 2 == 0) ? -zOffset : zOffset;
            float x2 = startX + j * spacing;
            float z2 = (j % 2 == 0) ? -zOffset : zOffset;
            DrawArrow(x1, y, z1, x2, y - LAYER_SPACING, z2);
        }
    }
}

// Render text at the given 3D position.
void RenderText(const char* text, float x, float y, float z) {
    glRasterPos3f(x, y, z);
    glPushAttrib(GL_LIST_BIT);
    glListBase(baseList - 32);
    glCallLists((GLsizei)strlen(text), GL_UNSIGNED_BYTE, text);
    glPopAttrib();
}

//-------------------------
// Network Drawing
//-------------------------

void DrawNetwork(void) {
    float prevY = TOP_Y;
    int first = 1;
    for (int i = 0; i < numLayers; i++) {
        float y = TOP_Y - i * LAYER_SPACING;
        // Draw arrow connecting centers of consecutive layers.
        if (!first) {
            DrawArrow(0.0f, prevY, 0.0f, 0.0f, y, 0.0f);
        } else {
            first = 0;
        }
        if (networkLayers[i].type == LAYER_BOX) {
            glColor3fv(networkLayers[i].color);
            DrawBox(0.0f, y, 0.0f,
                    networkLayers[i].box.width,
                    networkLayers[i].box.height,
                    networkLayers[i].box.depth);
        } else if (networkLayers[i].type == LAYER_FC) {
            glColor3fv(networkLayers[i].color);
            DrawFullyConnectedLayer(y, networkLayers[i].fc.neuronCount);
        }
        // Render label near the layer.
        glColor3f(1.0f, 1.0f, 1.0f); // White text
        RenderText(networkLayers[i].label, TEXT_OFFSET_X, y, 0.0f);
        prevY = y;
    }
}

//-------------------------
// Rendering & Window Handling
//-------------------------

void RenderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    // Set a top-down view.
    gluLookAt(0.0, 20.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, -1.0);
    glRotatef(rotX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotY, 0.0f, 1.0f, 0.0f);
    
    DrawNetwork();
    
    SwapBuffers(hDC);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch(message) {
        case WM_LBUTTONDOWN:
            mouseDown = 1;
            lastMouseX = LOWORD(lParam);
            lastMouseY = HIWORD(lParam);
            break;
        case WM_LBUTTONUP:
            mouseDown = 0;
            break;
        case WM_MOUSEMOVE:
            if (mouseDown) {
                int currentX = LOWORD(lParam);
                int currentY = HIWORD(lParam);
                int dx = currentX - lastMouseX;
                int dy = currentY - lastMouseY;
                rotY += dx * 0.5f;
                rotX += dy * 0.5f;
                lastMouseX = currentX;
                lastMouseY = currentY;
            }
            break;
        case WM_CLOSE:
            PostQuitMessage(0);
            break;
        case WM_DESTROY:
            return 0;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

//-------------------------
// Main Entry Point
//-------------------------

int WINAPI WinMain(HINSTANCE hInstanceCurrent, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    SetupConsole();
    SetupNetwork();
    
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstanceCurrent;
    wc.lpszClassName = "OpenGLWindowClass";
    if (!RegisterClass(&wc)) {
        MessageBox(NULL, "Failed to register window class.", "Error", MB_OK | MB_ICONERROR);
        return 0;
    }
    
    hInstance = hInstanceCurrent;
    hWnd = CreateWindow("OpenGLWindowClass", "3D Network Visualization", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                        100, 100, 800, 600, NULL, NULL, hInstance, NULL);
    if (!hWnd) {
        MessageBox(NULL, "Failed to create window.", "Error", MB_OK | MB_ICONERROR);
        return 0;
    }
    
    hDC = GetDC(hWnd);
    SetupPixelFormatForDC(hDC);
    hRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hRC);
    
    InitOpenGL();
    
    MSG msg;
    BOOL done = FALSE;
    while (!done) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                done = TRUE;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        RenderScene();
    }
    
    gluDeleteQuadric(quadric);
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hRC);
    ReleaseDC(hWnd, hDC);
    DestroyWindow(hWnd);
    return (int) msg.wParam;
}
