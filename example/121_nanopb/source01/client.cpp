#include <array>
#include <deque>
extern "C" {
#include "data.pb.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pb.h>
#include <pb_decode.h>
#include <pb_encode.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
};
#define FMT_HEADER_ONLY
#include "core.h"

bool read_callback(pb_istream_t *stream, uint8_t *buf, size_t count) {
  auto fd = reinterpret_cast<intptr_t>(stream->state);

  if (0 == count) {
    return true;
  }
  // operation should block until full request is satisfied. may still return
  // less than requested (upon signal, error or disconnect)

  auto result = recv(fd, buf, count, MSG_WAITALL);
  fmt::print("read_callback  count='{}'  result='{}'\n", count, result);
  for (auto i = 0; i < count; i += 1) {
    fmt::print("{:02x} ", buf[i]);
  }
  fmt::print("\n");
  if (0 == result) {
    // EOF
    stream->bytes_left = 0;
  }
  return count == result;
}

bool write_callback(pb_ostream_t *stream, const pb_byte_t *buf, size_t count) {
  auto fd = reinterpret_cast<intptr_t>(stream->state);

  for (auto i = 0; i < count; i += 1) {
    fmt::print("w  i='{}'  buf[i]='{}'\n", i, buf[i]);
  }
  return count == send(fd, buf, count, 0);
}

pb_istream_t pb_istream_from_socket(int fd) {
  auto stream = pb_istream_t(
      {.callback = read_callback,
       .state = reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
       .bytes_left = SIZE_MAX});
  return stream;
}

pb_ostream_t pb_ostream_from_socket(int fd) {
  auto stream = pb_ostream_t(
      {.callback = write_callback,
       .state = reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
       .max_size = SIZE_MAX,
       .bytes_written = 0});
  return stream;
}

void talk() {
  auto s = socket(AF_INET, SOCK_STREAM, 0);
  auto server_addr =
      sockaddr_in({.sin_family = AF_INET, .sin_port = htons(1234)});
  inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
  if (connect(s, reinterpret_cast<sockaddr *>(&server_addr),
              sizeof(server_addr))) {
    fmt::print("error connecting\n");
  }
  auto omsg = DataResponse({.temperature = (12.340f)});
  auto output = pb_ostream_from_socket(s);
  if (!(pb_encode(&output, DataResponse_fields, &omsg))) {
    fmt::print("error encoding\n");
  }
  // close the output stream of the socket, so that the server receives a FIN
  // packet
  shutdown(s, SHUT_WR);
  auto imsg = DataRequest({});
  auto input = pb_istream_from_socket(s);
  if (!(pb_decode(&input, DataRequest_fields, &imsg))) {
    fmt::print("error decoding\n");
  }
  fmt::print("  imsg.count='{}'  imsg.start_index='{}'\n", imsg.count,
             imsg.start_index);
}

int main(int argc, char **argv) {
  fmt::print("generation date 08:37:59 of Wednesday, 2023-04-12 (GMT+1)\n");
  talk();
  return 0;
}
