#include <deque>
extern "C" {
#include "data.pb.h"
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
    fmt::print("r  i='{}'  buf[i]='{}'\n", i, buf[i]);
  }
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

void handle_connection(int connfd) {
  auto input = pb_istream_from_socket(connfd);
  auto request = DataRequest();
  if (!(pb_decode(&input, DataRequest_fields, &request))) {
    fmt::print("error decode  PB_GET_ERROR(&input)='{}'\n",
               PB_GET_ERROR(&input));
  }
  fmt::print("request  request.count='{}'  request.start_index='{}'\n",
             request.count, request.start_index);
  auto response = DataResponse();
  auto output = pb_ostream_from_socket(connfd);
  response.index = 0;

  response.datetime = 123;

  response.pressure = (1234.50f);

  response.humidity = (47.30f);

  response.temperature = (23.20f);

  response.co2_concentration = (456.f);

  if (!(pb_encode(&output, DataResponse_fields, &response))) {
    fmt::print("error encoding\n");
  }
}

int main(int argc, char **argv) {
  fmt::print("generation date 23:38:53 of Monday, 2023-04-10 (GMT+1)\n");
  auto listenfd = socket(AF_INET, SOCK_STREAM, 0);
  auto reuse = int(1);
  setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  auto servaddr = sockaddr_in();
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;

  servaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  servaddr.sin_port = htons(1234);

  if (!(0 == bind(listenfd, reinterpret_cast<sockaddr *>(&servaddr),
                  sizeof(servaddr)))) {
    fmt::print("error bind\n");
  }
  if (!(0 == listen(listenfd, 5))) {
    fmt::print("error listen\n");
  }
  while (true) {
    auto connfd = accept(listenfd, nullptr, nullptr);
    if (connfd < 0) {
      fmt::print("error accept\n");
    }
    handle_connection(connfd);
    close(connfd);
  }

  return 0;
}
