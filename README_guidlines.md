- this document lists opinionated guidelines of how i want to write
  c++ to avoid foodguns
  
- 

## Variables




## Classes


- never use friends

### Constructors (1/2)

- always use initializer lists when possible

- use braces in initializer list to prevent implicit type conversions, e.g.:
```
class A {
public:
A(int n): value{n}{}
private:
int value;
};
```

- the order of initializers must be the same as the order of
  declaration

### Constructors (2/2)

- follow the rule of three or rule of five

**The Rule of Three**

The Rule of Three is a classic guideline for good resource management
in C++.  It states:

* **If your class needs an explicitly defined destructor, copy
  constructor, or copy assignment operator, it probably needs all
  three.**

The reason for this lies in how C++ handles object management by
default. If you don't define these special member functions, the
compiler will generate implicit versions for you. These implicit
versions usually do a shallow copy (member-by-member copy), which
works fine for simple classes.  However, problems arise when your
class manages resources like:

* **Dynamically allocated memory (pointers)**: A shallow copy might
  lead to multiple objects pointing to the same memory, causing
  double-deletion issues and memory corruption.

* **File handles, network connections, etc.**: Shallow copies can
  result in resources not being closed or connections not terminated
  properly.

**The Rule of Five**

The Rule of Five expands on the Rule of Three to encompass move
semantics introduced in C++11. It states:

* **If your class needs an explicitly defined destructor, copy
  constructor, copy assignment operator, move constructor, or move
  assignment operator, it likely needs all five.**

Move semantics provide a way to optimize object transfers by
"stealing" the resources of a temporary object instead of creating a
full, expensive copy. This is especially useful for large data
structures like containers.

**Key Takeaways**

* Adhering to the Rule of Three and Rule of Five is crucial for
  preventing memory leaks, resource mismanagement, and unexpected
  object behavior in C++.

* If your class deals with any form of "ownership" of resources,
  you'll likely need to consider these rules.
  
* Modern C++ often emphasizes the **Rule of Zero**, relying on smart
  pointers and standard library containers to automate resource
  management and avoid manual definitions of the special member
  functions.

## Member variables

- don't use prepend or append special characters to member
  variables. i don't want m_value or value_ to be the name of member
  variables. this is good code:
  
```
class A {
public:
	A() : value{0} {}
private:
    int value;
};
```

## Member methods

- declare methods const where possible (e.g. all getter methods):

```
class A {
public:
	A() : value{0} {}
	int get_value() const { return value };
private:
    int value;
};
```
