#include <vermell/vermell.h>

int main() {
  auto router = Router();
  router.setPort(8080);
  router.get("/", {[&](Query &http) { http.send("Hello from Vermell"); }});
  router.listen();
}
