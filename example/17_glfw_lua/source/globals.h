#ifndef GLOBALS_H

#define GLOBALS_H

#include <GLFW/glfw3.h>
extern "C" {
#include "lua/lua.h"
};

#include <complex>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "proto2.h"

#include <list>
// shapes for 2d cad
struct Shape;
struct Node {
  Shape *parent;
  glm::vec2 pos;
};
struct Shape {
  std::vector<Node> nodes;
  int max_nodes = 0;
  static float world_scale;
  static glm::vec2 world_offset;
  glm::vec4 color;
  virtual void draw() = 0;
  Node *hit_node(glm::vec2 &p) {
    // p is in world space
    for (auto &n : nodes) {
      if ((glm::distance(p, n.pos)) < ((1.00e-2))) {
        return &n;
      };
    };
    return nullptr;
  }
  void draw_nodes() {
    for (auto n : nodes) {
      auto sx = 0;
      auto sy = 0;
      world_to_screen(n.pos, sx, sy);
      glColor4f((1.0f), (0.30f), (0.30f), (1.0f));
      draw_circle(sx, sy, 2);
    };
  }
  void world_to_screen(const glm::vec2 &v, int &screeni, int &screenj) {
    screeni =
        static_cast<int>(((((v[0]) - (world_offset[0]))) * (world_scale)));
    screenj =
        static_cast<int>(((((v[1]) - (world_offset[1]))) * (world_scale)));
  }
  Node *get_next_node(const glm::vec2 &p) {
    if ((nodes.size()) == (max_nodes)) {
      return nullptr;
    };
    Node n;
    n.parent = this;
    n.pos = p;
    nodes.push_back(n);
    return &(nodes[((nodes.size()) - (1))]);
  }
};
struct Line : public Shape {
  Line() {
    max_nodes = 2;
    nodes.reserve(max_nodes);
    color = glm::vec4((1.0f), (1.0f), (0.f), (1.0f));
  };
  void draw() {
    auto sx = 0;
    auto sy = 0;
    auto ex = 0;
    auto ey = 0;
    world_to_screen(nodes[0].pos, sx, sy);
    world_to_screen(nodes[1].pos, ex, ey);
    glColor4f(color[0], color[1], color[2], color[3]);
    glBegin(GL_LINES);
    glVertex2i(sx, sy);
    glVertex2i(ex, ey);
    glEnd();
  }
};
struct Box : public Shape {
  Box() {
    max_nodes = 2;
    nodes.reserve(max_nodes);
    color = glm::vec4((1.0f), (1.0f), (0.f), (1.0f));
  };
  void draw() {
    auto sx = 0;
    auto sy = 0;
    auto ex = 0;
    auto ey = 0;
    world_to_screen(nodes[0].pos, sx, sy);
    world_to_screen(nodes[1].pos, ex, ey);
    glColor4f(color[0], color[1], color[2], color[3]);
    glBegin(GL_LINE_STRIP);
    glVertex2i(sx, sy);
    glVertex2i(ex, sy);
    glVertex2i(ex, ey);
    glVertex2i(sx, ey);
    glEnd();
  }
};
struct Circle : public Shape {
  Circle() {
    max_nodes = 2;
    nodes.reserve(max_nodes);
    color = glm::vec4((1.0f), (1.0f), (0.f), (1.0f));
  };
  void draw() {
    auto sx = 0;
    auto sy = 0;
    auto ex = 0;
    auto ey = 0;
    world_to_screen(nodes[0].pos, sx, sy);
    world_to_screen(nodes[1].pos, ex, ey);
    glColor4f(color[0], color[1], color[2], color[3]);
    glBegin(GL_LINES);
    glVertex2i(sx, sy);
    glVertex2i(ex, ey);
    glEnd();
    auto radius = ((world_scale) * (glm::distance(nodes[0].pos, nodes[1].pos)));
    draw_circle(sx, sy, radius);
  }
};
struct CommunicationTransaction {
  long long int start_loop_time;
  long long int tx_time;
  long long int rx_time;
  std::string tx_message;
  std::string rx_message;
};
typedef struct CommunicationTransaction CommunicationTransaction;
template <typename T, int MaxLen> class FixedDequeT : public std::deque<T> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  void push_back(const T &val) {
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<T>::push_back(val);
  }
};
template <typename T, int MaxLen> class FixedDequeTM : public std::deque<T> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  std::mutex mutex;
  void push_back(const T &val) {
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<T>::push_back(val);
  }
};
template <int MaxLen> class FixedDeque : public std::deque<float> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  void push_back(const float &val) {
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<float>::push_back(val);
  }
};

template <int MaxLen> class FixedGuardedDeque {
private:
  std::mutex mutex;
  FixedDeque<MaxLen> deque;

public:
  void push_back(const float &val) {
    {
      // no debug
      std::lock_guard<std::mutex> guard(mutex);
      deque.push_back(val);
    };
  }
  bool empty() {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return deque.empty();
  }
  size_t size() {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return deque.size();
  }
  float back() {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return deque.back();
  }
  float operator[](size_t n) {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return deque[n];
  }
};
template <int MaxLen> class FixedGuardedWaitingDeque {
  // https://baptiste-wicht.com/posts/2012/04/c11-concurrency-tutorial-advanced-locking-and-condition-variables.html
private:
  std::mutex mutex;
  std::condition_variable not_empty;
  FixedDeque<MaxLen> deque;

public:
  void push_back(const float &val) {
    {
      // no debug
      std::lock_guard<std::mutex> guard(mutex);
      deque.push_back(val);
    };
    not_empty.notify_one();
  }
  float back() {
    std::unique_lock<std::mutex> lk(mutex);
    while ((0) == (deque.size())) {
      not_empty.wait(lk);
    }
    return deque.back();
  }
  float operator[](size_t n) {
    std::unique_lock<std::mutex> lk(mutex);
    while ((0) == (deque.size())) {
      not_empty.wait(lk);
    }
    return deque[n];
  }
};
template <typename T, int MaxLen> class FixedQueue : public std::queue<T> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  void push(const T &val) {
    if ((MaxLen) == (this->size())) {
      this->pop();
    };
    std::queue<T>::push(val);
  }
};
template <typename T, int MaxLen> class GuardedWaitingQueue {
  // https://baptiste-wicht.com/posts/2012/04/c11-concurrency-tutorial-advanced-locking-and-condition-variables.html
private:
  std::mutex mutex;
  std::condition_variable not_empty;
  std::condition_variable not_full;
  std::queue<T> queue;

public:
  void push(const T &val) {
    {
      std::unique_lock<std::mutex> lk(mutex);
      while ((MaxLen) == (queue.size())) {
        not_full.wait(lk);
      }
      queue.push(val);
      lk.unlock();
    };
    not_empty.notify_one();
  }
  T front_no_wait() {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return queue.front();
  }
  size_t size_no_wait() {
    // no debug
    std::lock_guard<std::mutex> guard(mutex);
    return queue.size();
  }
  T front_and_pop() {
    // fixme: should this return a reference?
    std::unique_lock<std::mutex> lk(mutex);
    while ((0) == (queue.size())) {
      not_empty.wait(lk);
    }
    if (queue.size()) {
      auto result = queue.front();
      queue.pop();
      lk.unlock();
      not_full.notify_one();
      return result;
    } else {
      throw std::runtime_error("can't pop empty");
    }
  }
};

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  lua_State *_lua_state;
  bool _gui_request_diff_reset;
  std::mutex _gui_mutex;
  Node *_selected_node;
  std::list<Shape *> _shapes;
  Shape *_temp_shape;
  glm::vec2 _snapped_world_cursor;
  float _screen_grid;
  float _screen_scale;
  glm::vec2 _screen_start_pan;
  glm::vec2 _screen_offset;
  float _draw_marker_x;
  float _draw_alpha;
  float _draw_scale_y;
  float _draw_scale_x;
  float _draw_offset_y;
  float _draw_offset_x;
  bool _draw_display_log;
  std::mutex _draw_mutex;
  GLuint _fontTex;
  bool _framebufferResized;
  GLFWwindow *_window;
  double _cursor_ypos;
  double _cursor_xpos;
  std::string _code_generation_time;
  std::string _code_repository;
  std::string _main_version;
};
typedef struct State State;

#endif
