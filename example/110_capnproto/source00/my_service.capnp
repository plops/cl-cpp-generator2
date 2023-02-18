@0xdc35abf53c82d81e;

interface MyService {
  struct Params {
    arg1 @0 :Int32;
    arg2 @1 :Int32;
  }

  struct Results {
    result @0 :Int32;
  }

  calculate @0 (params :Params) -> (results :Results);
}