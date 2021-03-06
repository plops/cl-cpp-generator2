#ifndef PROTO2_H
#define PROTO2_H
void mainLoop ();
void run ();
int main ();
void keyCallback (GLFWwindow* window, int key, int scancode, int action, int mods);
void errorCallback (int err, const char* description);
static void framebufferResizeCallback (GLFWwindow* window, int width, int height);
void initWindow ();
void cleanupWindow ();
void uploadTex (const void* image, int w, int h);
int screen_width ();
int screen_height ();
glm::vec2 get_mouse_position ();
void draw_circle (float sx, float sy, float rad);
void initDraw ();
void world_to_screen (const glm::vec2 & v, int& screeni, int& screenj);
void screen_to_world (int screeni, int screenj, glm::vec2 & v);
void cleanupDraw ();
void drawFrame ();
void initGui ();
void cleanupGui ();
float get_FixedDeque (void* data, int idx);
void drawGui ();
bool checkLua (lua_State* L, int res);
int lua_HostFunction (lua_State* L);
void initLua ();
void cleanupLua ();
#endif