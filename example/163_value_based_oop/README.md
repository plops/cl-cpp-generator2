# Introduction

Here I'm learning about type erasure. This is a programming pattern
that helps with decoupling dependencies. By this two problems are avoided:
1) Lots of derived classes that need to be updated whenever the a strategy is changed.
2) Header-only libraries (and slow the resulting slow compilation times) because of many templates

A diagram that shows what kind of issues this solves is shown in this
short talk:
'MUC++_Klaus_Iglberger_-_A_Brief_Introduction_to_Type_Erasure-[SPelQUPcHQQ]'
at minute 8:00.

Note that in main{,2,3}.cpp the Interface and Implementation classes
shall be private in StatelessTE, StatefulTE and UniversalTE,
respectively.


source02 has interface and implementation that are hidden by pimpl
in source01 these are called concept and model, respectively (which apparently is the naming convention in programming pattern literature)


|       |                                                                                   |
| gen04 | use lisp to define the UniversalTE class                                          |
| gen05 | try to figure out how to declare internal classes with implementation in cpp file |
|       |                                                                                   |

# Where to declare the private classes?

According to cpp design chapter "The pimpl idiom" forward declare
class Implementation in the header file and define the class in the
cpp file.

# How to deal with the unique_ptr pimpl?

According to cpp design chapter "The pimpl idiom".  We need to define
the destructor of the class that contains the pimpl unique_ptr in the
cpp file:

```
UniversalTE::~UniversalTE() = default;
```

Otherwise the compiler will call the destructor of the pimpl
unique_ptr, crossing the information boundary that the bridge design
pattern is supposed to uphold.


The unique_ptr can't be copied, so in order for the UniversalTE class
to be useful (copyable) we need to implement according to the rule of
5. note that the pimpl will not change after initialization and is
   therefore declared as const:
   
```
std::unique_ptr<Interface> const pimpl;
```

Note: This const doesn't work with UniversalTE class

## Copy constructor 17

```
UniversalTE::UniversalTE( UniversalTE const& other )
  : pimpl{ make_unique<Interface>(*other.pimpl) }
{}
```

## Copy assignment operator 18

```
UniversalTE& UniversalTE::operator=( UniversalTE const& other )
{
	*pimpl = *other.pimpl;
	return *this;
}
```

## Move constructor 19

allocates new memory with make_unique(), which may fail or throw
therefore the move constructor is not noexcept

```
UniversalTE::UniversalTE( UniversalTE&& other )
  : pimpl{ make_unique<Interface>( std::move(*other.pimpl)) }
{}
```

## Move assignment operator 20
```
UniversalTE& UniversalTE::operator=( UniversalTE&& other )
{
  *pimpl = std::move(*other.pimpl);
  return *this;
}
```


# Notes

Visitor:
  - works for open set of operations (you can easily add more)
Strategy: configuration details are passed (dependency injection) from the outside (behavioural) (guidelines 23 and )
  - works for open set of types (orthogonal to visitor)
  - one strategy deals with one operation (e.g. drawing but not serialization)
    - for more operations look at `external polymorphism` and `type erasure`
Bridge: class knows about details but wants to reduce dependencies on these details (structural)
Prototype: provides virtual clone function (works with open set of types) (guideline 30)

External Polymorphism (p. 3932)


# References

## Youtube video 1 
- There_is_no_silver_bullet_-_Klaus_Iglberger_-_Meeting_C++_2024
- 5F3d3LocM3k

keywords: value-based object oriented programming, Progressive C++,
type erasure

**No Silver Bullet: Exploring Design Choices in Modern C++**

*   **0:00:08 Introduction:** Claus introduces the topic of design
    choices in modern C++, focusing on the shift away from traditional
    object-oriented patterns.
*   **0:00:33 The Problem with Inheritance:** The traditional approach
    using inheritance and virtual functions leads to tight coupling
    and maintenance issues, especially in large codebases.
*   **0:09:09 Modern C++ Isn't Always the Answer:** Modern C++
    features like unique pointers and move semantics don't
    automatically solve design problems associated with inheritance.
*   **0:13:19 Variant-Based Solution:** A variant-based approach
    offers a more functional style, eliminating inheritance and
    pointers for improved simplicity and potential performance gains.
*   **0:22:29 Architectural Issues with Variants:** Using variants
    directly can lead to tight coupling and hinder code
    maintainability, as it exposes implementation details and
    complicates extensibility.
*   **0:25:30 Templates as a Solution:** Function templates can help
    invert dependencies and improve code modularity when working with
    variants.
*   **0:27:36 The Template Trap:** Overusing templates can result in
    excessive code bloat and increased compile times, especially in
    large projects.
*   **0:28:32 The Spectrum of Design Choices:** Variants and virtual
    functions represent opposite ends of a design spectrum, each with
    its own strengths and weaknesses.
*   **0:30:17 The Importance of Design:** Good software design, rather
    than specific language features, is crucial for project
    success. The talk emphasizes the need for a stronger focus on
    design principles in the C++ community.
*   **0:36:26 Progressive C++:** The concept of "Progressive C++" is
    introduced as a new term to describe modern C++ development that
    emphasizes design philosophy over specific language features.
*   **0:36:38 Type Erasure as a Solution:** Type erasure, using a
    combination of inheritance and templates hidden behind a
    value-like interface, offers a balance between flexibility and
    maintainability.
*   **0:47:25 Performance Considerations:** The type erasure approach
    can offer performance benefits, especially when combined with
    small buffer optimizations.
*   **0:49:16 Choosing the Right Tool:** The choice between a
    variant-based (functional) and a type erasure-based
    (object-oriented) approach depends on the specific architectural
    needs of the project.
*   **0:51:51 Reflection and the Future:** C++ reflection, expected in
    future standards, may offer new possibilities for generating type
    erasure wrappers and potentially simplifying or replacing
    variant-based solutions.
*   **0:53:24 Conclusion:** There is no one-size-fits-all solution in
    software design. Understanding the trade-offs between different
    approaches and prioritizing architectural considerations are
    essential for building maintainable and efficient software.
*   **0:54:16 Q&A:** Using libraries to reduce boilerplate from type
    erasure and using C++ concepts with a functional approach are
    discussed.
*   **0:56:43 CRTP is not a replacement:** CRTP is not a general
    solution for replacing inheritance hierarchies due to its
    limitations in terms of type relationships and its tendency to
    push designs towards a template-heavy approach.

## Youtube Video 2

MUC++_Nicolas_Chausse_-_Two_Kinds_of_Type_Erasure-[QgMQPqVc6JE]
