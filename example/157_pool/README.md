CppCon 2017： Pablo Halpern “Allocators： The Good Parts” [v3dz-AKOVL8]

32:08 Test resource 


C++17's std::pmr comes with a cost

synchronized_pool_resource .. for data frequently accessed at the same time
  data in same place in memory
  
unsynchronized_pool_resoruce .. no thread sync cost
  if only one thread is using the allocator

monotonic_buffer_resource .. high speed, high space
  if your access patterns fit this, it is wonderful to use
  
LoggingResource

 .. typically you write a transformer (call the underlying resource)
 
 
make resource static
another static vector (defined in another function) might get an element that was allocated with our allocator

when main closes, the memoryResource is destroyed before the thing that uses is (a static variable) gets destroyed

note: you may never destroy a memory resource before everything that uses is has been deallocated

you can't set default allocator for variables you allocate before main


if you don't do the right thing you get the wrong behaviour

unique_ptr must call polymorphic_allocator_delete (you must not call the normal delete)
this makes the unique_ptr bigger

you can set the default allocator in main, then the correct delete will be called

try 
 construct
and 
 deallocate
 throw
 
destroy
deallocate


if a move constructor can throw an exception
that is why the std:vector will copy (so that it can undo)

dirty little secret (you need to have that otherwise you have terrible performance in std vector)
you create a move constructor that copies the allocator

how to prevent all the same pointers to allocators?
you use a pointer and not a unique_ptr

get_allocator() accessor to reduce redundant copies and avoid bloated object

you need a lot of code and tests! i don't think this is worth it


alternatives:

object pools
custom datatypes


tcmalloc allows you to debug allocations (but this is orthogonal to pmr)

suggested talk about the benefits of allocators: local arena memory allocators
meeting c++ november 2017 in berlin

