double now ();
void mainLoop ();
void run ();
int main ();
size_t get_filesize (const char* filename);
void destroy_mmap ();
void init_mmap (const char* filename);
void keyCallback (GLFWwindow* window, int key, int scancode, int action, int mods);
void errorCallback (int err, const char* description);
static void framebufferResizeCallback (GLFWwindow* window, int width, int height);
void initWindow ();
void cleanupWindow ();
